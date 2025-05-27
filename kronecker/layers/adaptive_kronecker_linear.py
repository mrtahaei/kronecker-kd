import torch
import torch.nn as nn
import math
import time
from typing import Optional, Tuple, Literal
import warnings

# Import the two implementations with absolute imports
try:
    from kronecker_triton_kernel_2 import TritonKroneckerLinear, kron_linear_forward
except ImportError:
    try:
        from .kronecker_triton_kernel_2 import TritonKroneckerLinear, kron_linear_forward
    except ImportError:
        # Fallback - define dummy functions
        def kron_linear_forward(*args, **kwargs):
            raise ImportError("Triton implementation not available")
        TritonKroneckerLinear = None

try:
    from kronecker_linear_optimized import KroneckerLinear as OptimizedKroneckerLinear, efficient_kronecker_linear
except ImportError:
    try:
        from .kronecker_linear_optimized import KroneckerLinear as OptimizedKroneckerLinear, efficient_kronecker_linear
    except ImportError:
        # Fallback - define dummy functions
        def efficient_kronecker_linear(*args, **kwargs):
            raise ImportError("Optimized implementation not available")
        OptimizedKroneckerLinear = None


class AdaptiveKroneckerLinear(nn.Module):
    """
    Adaptive Kronecker Linear layer that automatically chooses the best implementation
    (Triton kernel vs optimized PyTorch) based on a warmup benchmark during initialization.
    
    This class provides a unified interface while automatically selecting the most efficient
    backend for the given configuration and hardware.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_factors: Optional[Tuple[int, int]] = None,
        out_factors: Optional[Tuple[int, int]] = None,
        bias: bool = True,
        device=None,
        dtype=None,
        warmup_samples: int = 10,
        warmup_batch_size: int = 32,
        force_implementation: Optional[Literal["triton", "optimized"]] = None,
        verbose: bool = False
    ):
        """
        Initialize the adaptive Kronecker linear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension  
            in_factors: Factorization of input dimension (factor1, factor2)
            out_factors: Factorization of output dimension (factor1, factor2)
            bias: Whether to include bias term
            device: Device to place tensors on
            dtype: Data type for tensors
            warmup_samples: Number of forward passes for benchmarking
            warmup_batch_size: Batch size for warmup benchmark
            force_implementation: Force a specific implementation ("triton" or "optimized")
            verbose: Print benchmark results
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.warmup_samples = warmup_samples
        self.warmup_batch_size = warmup_batch_size
        self.verbose = verbose
        
        # Determine optimal factors if not provided
        if in_factors is None:
            in_factors = self._get_optimal_factors(in_features)
        if out_factors is None:
            out_factors = self._get_optimal_factors(out_features)
            
        # Validate factors
        if in_factors[0] * in_factors[1] != in_features:
            raise ValueError(f"in_factors {in_factors} must multiply to in_features {in_features}")
        if out_factors[0] * out_factors[1] != out_features:
            raise ValueError(f"out_factors {out_factors} must multiply to out_features {out_features}")
            
        self.in_factors = in_factors
        self.out_factors = out_factors
        
        # Store factor names for convenience
        self.in_factor1, self.in_factor2 = in_factors
        self.out_factor1, self.out_factor2 = out_factors
        
        # Set device and dtype
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.dtype = dtype or torch.float32
        
        # Initialize parameters
        self.kn_factor_A = nn.Parameter(torch.empty((self.out_factor1, self.in_factor1), device=device, dtype=dtype))
        self.kn_factor_B = nn.Parameter(torch.empty((self.out_factor2, self.in_factor2), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        
        self.reset_parameters()
        
        # Choose implementation
        if force_implementation:
            # Validate constraints for forced implementation
            if force_implementation == "triton":
                # Check if Triton is available and constraints are met
                triton_available = True
                try:
                    import triton
                except ImportError:
                    triton_available = False
                
                if not triton_available:
                    if verbose:
                        print("Warning: Triton not available, falling back to optimized implementation")
                    self.implementation = "optimized"
                elif not torch.cuda.is_available():
                    if verbose:
                        print("Warning: CUDA not available, falling back to optimized implementation")
                    self.implementation = "optimized"
                elif self.in_factor2 > 16:
                    if verbose:
                        print(f"Warning: B factor ({self.in_factor2}) too large for Triton kernel, falling back to optimized implementation")
                    self.implementation = "optimized"
                else:
                    self.implementation = force_implementation
                    if verbose:
                        print(f"Forced implementation: {self.implementation}")
            else:
                self.implementation = force_implementation
                if verbose:
                    print(f"Forced implementation: {self.implementation}")
        else:
            self.implementation = self._benchmark_implementations()
            
        # Set the forward function based on chosen implementation
        self._set_forward_function()
    
    @staticmethod
    def _get_optimal_factors(n: int) -> Tuple[int, int]:
        """Find factor pairs near the square root of n for balanced computation."""
        factor = int(math.sqrt(n))
        while factor > 1 and n % factor != 0:
            factor -= 1
        return (n // factor, factor)
    
    def reset_parameters(self):
        """Initialize the Kronecker factors using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.kn_factor_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.kn_factor_B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _benchmark_implementations(self) -> str:
        """
        Benchmark both implementations and choose the faster one.
        
        Returns:
            "triton" or "optimized" based on which is faster
        """
        if not torch.cuda.is_available():
            if self.verbose:
                print("CUDA not available, using optimized implementation")
            return "optimized"
        
        # Check if Triton is available and B factor is small enough
        try:
            import triton
            if self.in_factor2 > 16:  # Triton kernel constraint
                if self.verbose:
                    print(f"B factor ({self.in_factor2}) too large for Triton kernel, using optimized implementation")
                return "optimized"
        except ImportError:
            if self.verbose:
                print("Triton not available, using optimized implementation")
            return "optimized"
        
        # Create test input
        test_input = torch.randn(
            self.warmup_batch_size, 
            self.in_features, 
            device=self.device, 
            dtype=self.dtype
        )
        
        # Benchmark Triton implementation
        triton_time = float('inf')
        try:
            # Warmup
            for _ in range(3):
                _ = kron_linear_forward(test_input, self.kn_factor_A, self.kn_factor_B, self.bias)
            
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(self.warmup_samples):
                _ = kron_linear_forward(test_input, self.kn_factor_A, self.kn_factor_B, self.bias)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / self.warmup_samples
            
        except Exception as e:
            if self.verbose:
                print(f"Triton implementation failed: {e}")
            triton_time = float('inf')
        
        # Benchmark optimized implementation
        optimized_time = float('inf')
        try:
            # Warmup
            for _ in range(3):
                _ = efficient_kronecker_linear(test_input, self.kn_factor_A, self.kn_factor_B, self.bias)
            
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(self.warmup_samples):
                _ = efficient_kronecker_linear(test_input, self.kn_factor_A, self.kn_factor_B, self.bias)
            torch.cuda.synchronize()
            optimized_time = (time.time() - start_time) / self.warmup_samples
            
        except Exception as e:
            if self.verbose:
                print(f"Optimized implementation failed: {e}")
            optimized_time = float('inf')
        
        # Choose the faster implementation
        if triton_time < optimized_time:
            chosen = "triton"
            speedup = optimized_time / triton_time
        else:
            chosen = "optimized"
            speedup = triton_time / optimized_time
        
        if self.verbose:
            print(f"Benchmark results:")
            print(f"  Triton time: {triton_time:.6f}s")
            print(f"  Optimized time: {optimized_time:.6f}s")
            print(f"  Chosen: {chosen} (speedup: {speedup:.2f}x)")
        
        return chosen
    
    def _set_forward_function(self):
        """Set the forward function based on the chosen implementation."""
        if self.implementation == "triton":
            self._forward_impl = self._triton_forward
        else:
            self._forward_impl = self._optimized_forward
    
    def _triton_forward(self, x):
        """Forward pass using Triton kernel implementation."""
        return kron_linear_forward(x, self.kn_factor_A, self.kn_factor_B, self.bias)
    
    def _optimized_forward(self, x):
        """Forward pass using optimized PyTorch implementation."""
        return efficient_kronecker_linear(x, self.kn_factor_A, self.kn_factor_B, self.bias)
    
    def forward(self, x):
        """Forward pass using the chosen implementation."""
        # Ensure input is on the correct device
        if x.device != self.device:
            if self.verbose:
                warnings.warn(f"Input tensor moved from {x.device} to {self.device}")
            x = x.to(self.device)
        
        return self._forward_impl(x)
    
    def get_implementation_info(self) -> dict:
        """Get information about the chosen implementation."""
        return {
            "implementation": self.implementation,
            "in_factors": self.in_factors,
            "out_factors": self.out_factors,
            "device": self.device,
            "dtype": self.dtype,
            "parameters": self.kn_factor_A.numel() + self.kn_factor_B.numel() + (self.bias.numel() if self.bias is not None else 0),
            "original_parameters": self.in_features * self.out_features + (self.out_features if self.bias is not None else 0),
        }
    
    def switch_implementation(self, implementation: Literal["triton", "optimized"], verbose: bool = None):
        """
        Manually switch to a different implementation.
        
        Args:
            implementation: "triton" or "optimized"
            verbose: Override verbose setting for this operation
        """
        if verbose is None:
            verbose = self.verbose
            
        old_impl = self.implementation
        self.implementation = implementation
        self._set_forward_function()
        
        if verbose:
            print(f"Switched implementation from {old_impl} to {implementation}")
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        in_factors: Optional[Tuple[int, int]] = None,
        out_factors: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> 'AdaptiveKroneckerLinear':
        """
        Create an AdaptiveKroneckerLinear from an existing nn.Linear layer.
        
        Args:
            linear: The nn.Linear layer to convert
            in_factors: Input factorization (if None, will be computed)
            out_factors: Output factorization (if None, will be computed)
            **kwargs: Additional arguments for AdaptiveKroneckerLinear
        """
        # Create the adaptive layer
        adaptive_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            in_factors=in_factors,
            out_factors=out_factors,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            **kwargs
        )
        
        # Initialize from the linear layer using SVD decomposition
        W = linear.weight.data
        kn_A, kn_B = adaptive_layer._svd_nearest_kron(
            W, 
            adaptive_layer.out_factor1, 
            adaptive_layer.in_factor1,
            adaptive_layer.out_factor2, 
            adaptive_layer.in_factor2
        )
        
        adaptive_layer.kn_factor_A.data.copy_(kn_A)
        adaptive_layer.kn_factor_B.data.copy_(kn_B)
        
        if linear.bias is not None:
            adaptive_layer.bias.data.copy_(linear.bias.data)
        
        return adaptive_layer
    
    @staticmethod
    def _svd_nearest_kron(weight: torch.Tensor, m1: int, n1: int, m2: int, n2: int):
        """
        Compute the nearest Kronecker product decomposition via SVD.
        
        Args:
            weight: Weight matrix of shape (m1*m2, n1*n2)
            m1, n1: Dimensions for factor A
            m2, n2: Dimensions for factor B
            
        Returns:
            Tuple (A, B) where A is (m1, n1) and B is (m2, n2)
        """
        # Rearrange weight from (m1*m2, n1*n2) to (m1*n1, m2*n2)
        R = weight.view(m1, m2, n1, n2).permute(0, 2, 1, 3).contiguous().view(m1 * n1, m2 * n2)
        
        # Convert to float32 for SVD, then back to original dtype
        orig_dtype = R.dtype
        R_f32 = R.to(torch.float32)
        U, S, Vh = torch.linalg.svd(R_f32, full_matrices=False)
        U, S, Vh = U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)
        
        # Use the largest singular value component
        scale = math.sqrt(S[0].item())
        A = scale * U[:, 0].view(m1, n1)
        B = scale * Vh[0, :].view(m2, n2)
        
        return A, B
    
    def extra_repr(self) -> str:
        """Return a string representation of the module for debugging."""
        info = self.get_implementation_info()
        reduction = info["original_parameters"] / info["parameters"] if info["parameters"] > 0 else float('inf')
        
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'in_factors={self.in_factors}, out_factors={self.out_factors}, '
            f'implementation={self.implementation}, '
            f'bias={self.bias is not None}, params_reduction={reduction:.2f}x'
        )


# Convenience function for easy usage
def create_adaptive_kronecker_linear(
    in_features: int,
    out_features: int,
    **kwargs
) -> AdaptiveKroneckerLinear:
    """
    Convenience function to create an AdaptiveKroneckerLinear layer.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        **kwargs: Additional arguments for AdaptiveKroneckerLinear
        
    Returns:
        AdaptiveKroneckerLinear instance
    """
    return AdaptiveKroneckerLinear(in_features, out_features, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing AdaptiveKroneckerLinear...")
    
    # Test with different configurations
    configs = [
        # 2x2 B matrices
        (4096, 4096, (2048, 2), (2048, 2)),  # 2048x2 A and 2x2 B matrices
        (8192, 8192, (4096, 2), (4096, 2)),  # 4096x2 A and 2x2 B matrices
        (16384, 16384, (8192, 2), (8192, 2)), # 8192x2 A and 2x2 B matrices
        (32768, 32768, (16384, 2), (16384, 2)), # 16384x2 A and 2x2 B matrices
        
        # 4x4 B matrices
        (4096, 4096, (1024, 4), (1024, 4)),  # 1024x4 A and 4x4 B matrices
        (8192, 8192, (2048, 4), (2048, 4)),  # 2048x4 A and 4x4 B matrices
        (16384, 16384, (4096, 4), (4096, 4)), # 4096x4 A and 4x4 B matrices
        (32768, 32768, (8192, 4), (8192, 4)), # 8192x4 A and 4x4 B matrices
        
        # 8x8 B matrices
        (4096, 4096, (512, 8), (512, 8)),    # 512x8 A and 8x8 B matrices
        (8192, 8192, (1024, 8), (1024, 8)),  # 1024x8 A and 8x8 B matrices
        (16384, 16384, (2048, 8), (2048, 8)), # 2048x8 A and 8x8 B matrices
        (32768, 32768, (4096, 8), (4096, 8))  # 4096x8 A and 8x8 B matrices
    ]

    results = []
    for in_feat, out_feat, in_factors, out_factors in configs:
        print("\n" + "-"*80)
        print(f"{'Size':15} | {'A Matrix':15} | {'B Matrix':15} | {'Shapes':25} | {'Implementation'}")
        print("-"*80)
        
        # Test adaptive implementation
        layer_adaptive = AdaptiveKroneckerLinear(
            in_features=in_feat,
            out_features=out_feat,
            in_factors=in_factors,
            out_factors=out_factors,
            verbose=False,
            warmup_samples=100
        )
        
        # Force Triton implementation
        layer_triton = AdaptiveKroneckerLinear(
            in_features=in_feat,
            out_features=out_feat,
            in_factors=in_factors,
            out_factors=out_factors,
            force_implementation='triton',
            verbose=False
        )
        
        # Force PyTorch implementation
        layer_pytorch = AdaptiveKroneckerLinear(
            in_features=in_feat,
            out_features=out_feat,
            in_factors=in_factors,
            out_factors=out_factors,
            force_implementation='optimized',
            verbose=False
        )

        # Copy weights from adaptive layer to ensure consistency
        with torch.no_grad():
            layer_triton.kn_factor_A.copy_(layer_adaptive.kn_factor_A)
            layer_triton.kn_factor_B.copy_(layer_adaptive.kn_factor_B)
            if layer_triton.bias is not None:
                layer_triton.bias.copy_(layer_adaptive.bias)
                
            layer_pytorch.kn_factor_A.copy_(layer_adaptive.kn_factor_A)
            layer_pytorch.kn_factor_B.copy_(layer_adaptive.kn_factor_B)
            if layer_pytorch.bias is not None:
                layer_pytorch.bias.copy_(layer_adaptive.bias)

        # Create standard linear layer with kronecker product weight
        layer_standard = nn.Linear(in_feat, out_feat, device=layer_adaptive.device)
        with torch.no_grad():
            # Set weight to kronecker product of A and B
            kronecker_weight = torch.kron(layer_adaptive.kn_factor_A, layer_adaptive.kn_factor_B)
            layer_standard.weight.copy_(kronecker_weight)
            if layer_standard.bias is not None:
                layer_standard.bias.copy_(layer_adaptive.bias)

        # Test forward pass
        x = torch.randn(32, in_feat, device=layer_adaptive.device)
        
        # Time each implementation
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Warmup adaptive
        for _ in range(10):
            _ = layer_adaptive(x)
        torch.cuda.synchronize()
        
        # Time adaptive
        start.record()
        for _ in range(100):
            y_adaptive = layer_adaptive(x)
        end.record()
        torch.cuda.synchronize()
        time_adaptive = start.elapsed_time(end) / 100  # Average time per step
        
        # Warmup Triton
        for _ in range(10):
            _ = layer_triton(x)
        torch.cuda.synchronize()
        
        # Time Triton
        start.record()
        for _ in range(100):
            y_triton = layer_triton(x)
        end.record()
        torch.cuda.synchronize()
        time_triton = start.elapsed_time(end) / 100  # Average time per step
        
        # Warmup PyTorch
        for _ in range(10):
            _ = layer_pytorch(x)
        torch.cuda.synchronize()
        
        # Time PyTorch
        start.record()
        for _ in range(100):
            y_pytorch = layer_pytorch(x)
        end.record()
        torch.cuda.synchronize()
        time_pytorch = start.elapsed_time(end) / 100  # Average time per step

        # Warmup standard
        for _ in range(10):
            _ = layer_standard(x)
        torch.cuda.synchronize()
        
        # Time standard
        start.record()
        for _ in range(100):
            y_standard = layer_standard(x)
        end.record()
        torch.cuda.synchronize()
        time_standard = start.elapsed_time(end) / 100  # Average time per step

        # Verify outputs match
        if not torch.allclose(y_adaptive, y_standard, rtol=1e-3, atol=1e-3):
            max_diff = torch.max(torch.abs(y_adaptive - y_standard))
            print(f"Max difference between adaptive and standard: {max_diff}")
            print(f"Configuration: {in_feat} -> {out_feat}, in_factors: {in_factors}, out_factors: {out_factors}")
            print(f"B_K = {in_factors[1]}, chosen implementation: {layer_adaptive.implementation}")
            assert False, f"Adaptive output mismatch, max diff: {max_diff}"
            
        if not torch.allclose(y_triton, y_standard, rtol=1e-3, atol=1e-3):
            max_diff = torch.max(torch.abs(y_triton - y_standard))
            print(f"Max difference between triton and standard: {max_diff}")
            print(f"Configuration: {in_feat} -> {out_feat}, in_factors: {in_factors}, out_factors: {out_factors}")
            print(f"B_K = {in_factors[1]}, triton implementation: {layer_triton.implementation}")
            assert False, f"Triton output mismatch, max diff: {max_diff}"
            
        if not torch.allclose(y_pytorch, y_standard, rtol=1e-3, atol=1e-3):
            max_diff = torch.max(torch.abs(y_pytorch - y_standard))
            print(f"Max difference between pytorch and standard: {max_diff}")
            print(f"Configuration: {in_feat} -> {out_feat}, in_factors: {in_factors}, out_factors: {out_factors}")
            print(f"B_K = {in_factors[1]}")
            assert False, f"PyTorch output mismatch, max diff: {max_diff}"

        size_str = f"{in_feat} -> {out_feat}"
        a_matrix = f"{in_factors[0]}x{in_factors[1]}"
        b_matrix = f"{out_factors[0]}x{out_factors[1]}"
        shapes = f"{x.shape} -> {y_adaptive.shape}"
        impl = layer_adaptive.get_implementation_info()
        
        results.append({
            'config': f"{size_str} ({a_matrix} x {b_matrix})",
            'adaptive_time': time_adaptive,
            'triton_time': time_triton,
            'pytorch_time': time_pytorch,
            'standard_time': time_standard,
            'speedup_vs_triton': time_triton / time_adaptive,
            'speedup_vs_pytorch': time_pytorch / time_adaptive,
            'speedup_vs_standard': time_standard / time_adaptive,
            'chosen_impl': impl['implementation']
        })
        
        print(f"{size_str:15} | {a_matrix:15} | {b_matrix:15} | {shapes:25} | {impl}")
        print("-"*80)

    print("\nPerformance Summary:")
    print("="*120)
    print(f"{'Configuration':30} | {'Chosen':10} | {'vs Triton':15} | {'vs PyTorch':15} | {'vs Standard':15} | {'Times (ms)':30}")
    print("="*120)
    
    for result in results:
        print(f"{result['config']:30} | "
              f"{result['chosen_impl']:10} | "
              f"{result['speedup_vs_triton']:>6.2f}x faster | "
              f"{result['speedup_vs_pytorch']:>6.2f}x faster | "
              f"{result['speedup_vs_standard']:>6.2f}x faster | "
              f"A:{result['adaptive_time']:6.2f} T:{result['triton_time']:6.2f} P:{result['pytorch_time']:6.2f} S:{result['standard_time']:6.2f}")