import triton
import triton.language as tl
import torch
import math
from typing import Optional, Tuple
import torch.nn as nn


def efficient_kronecker_linear(x, a_kron_factor, b_kron_factor, bias):
    """
    Highly optimized Kronecker linear using the mathematical identity:
    Y = X @ (A ⊗ B)^T
    
    Instead of materializing the Kronecker product, we use:
    (A ⊗ B) @ vec(X) = vec(B @ X @ A^T)
    
    For matrix form: Y = X @ (A ⊗ B)^T
    We reshape X to (M, A_K, B_K), then:
    1. Apply B^T: temp = X @ B^T  -> (M, A_K, B_N)
    2. Apply A^T: Y = temp @ A^T  -> (M, A_N * B_N)
    
    This is much faster as it avoids creating the full Kronecker product!
    """
    M, K = x.shape
    A_N, A_K = a_kron_factor.shape
    B_N, B_K = b_kron_factor.shape
    
    assert K == A_K * B_K, f"Input dimension mismatch: {K} != {A_K} * {B_K}"
    
    # Reshape input to separate A and B dimensions
    x_reshaped = x.view(M, A_K, B_K)  # (M, A_K, B_K)
    
    # Step 1: Apply B^T to the last dimension using optimized matmul
    # x_reshaped: (M, A_K, B_K), B^T: (B_N, B_K)
    # Result: (M, A_K, B_N)
    temp = torch.matmul(x_reshaped, b_kron_factor.t())  # (M, A_K, B_N)
    
    # Step 2: Apply A^T to the middle dimension using optimized matmul
    # temp: (M, A_K, B_N), A^T: (A_N, A_K)
    # We need to contract over the A_K dimension
    # temp.permute(0, 2, 1): (M, B_N, A_K)
    # A^T: (A_N, A_K)
    # Result: (M, B_N, A_N)
    temp_perm = temp.permute(0, 2, 1)  # (M, B_N, A_K)
    result_perm = torch.matmul(temp_perm, a_kron_factor.t())  # (M, B_N, A_N)
    
    # Reshape to (M, A_N * B_N)
    # We need to interleave A_N and B_N properly to match Kronecker product layout
    result = result_perm.permute(0, 2, 1).contiguous().view(M, A_N * B_N)
    
    # Add bias
    if bias is not None:
        result = result + bias
    
    return result


class TritonKroneckerLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_kron_factor, b_kron_factor, bias):
        ctx.save_for_backward(x, a_kron_factor, b_kron_factor)
        return efficient_kronecker_linear(x, a_kron_factor, b_kron_factor, bias)
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        Memory‑friendly backward pass using the same mathematical identity
        """
        x, A, B = ctx.saved_tensors           # A: (A_N, A_K)   B: (B_N, B_K)
        M, K      = x.shape
        A_N, A_K  = A.shape
        B_N, B_K  = B.shape
        assert K == A_K * B_K, "K mismatch"

        # ---------- grad_x -------------------------------------------------
        # 1. shape grad_out -> (M, A_N, B_N)
        gout_mpq = grad_out.view(M, A_N, B_N)          # m p q
        # 2. contract over p with A  (A_N, A_K)
        #       m p q , p k  -> m k q
        tmp_mkq  = torch.einsum('mpq,pk->mkq', gout_mpq, A)  # (M, A_K, B_N)
        # 3. contract over q with B  (B_N, B_K)
        #       m k q , q l  -> m k l
        x_grad_mkl = torch.einsum('mkq,ql->mkl', tmp_mkq, B) # (M, A_K, B_K)
        grad_x = x_grad_mkl.reshape(M, K)                    # (M, K)

        # ---------- grad_A -------------------------------------------------
        x_mkl = x.view(M, A_K, B_K)                     # m k l
        grad_A = torch.einsum('mkl,mpq,ql->pk', x_mkl, gout_mpq, B)  # (A_N, A_K)

        # ---------- grad_B -------------------------------------------------
        grad_B = torch.einsum('mkl,mpq,pk->ql', x_mkl, gout_mpq, A)  # (B_N, B_K)

        # ---------- grad_bias ---------------------------------------------
        grad_bias = grad_out.sum(dim=0)

        return grad_x, grad_A, grad_B, grad_bias


class KroneckerLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_factors: Optional[Tuple[int, int]] = None,
        out_factors: Optional[Tuple[int, int]] = None,
        bias: bool = True,
        device=None,
        dtype=None
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Validate that factors multiply to the correct dimensions
        if in_factors[0] * in_factors[1] != in_features:
            raise ValueError(f"in_factors {in_factors} must multiply to in_features {in_features}")
        if out_factors[0] * out_factors[1] != out_features:
            raise ValueError(f"out_factors {out_factors} must multiply to out_features {out_features}")
        
        # Descriptive factor naming
        self.in_factor1, self.in_factor2 = in_factors
        self.out_factor1, self.out_factor2 = out_factors

        # Initialize Kronecker factors; shapes: (out_factor, in_factor)
        self.kn_factor_A = nn.Parameter(torch.empty((self.out_factor1, self.in_factor1), device=device, dtype=dtype))
        self.kn_factor_B = nn.Parameter(torch.empty((self.out_factor2, self.in_factor2), device=device, dtype=dtype))
        # Bias parameter
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the Kronecker factors using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.kn_factor_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.kn_factor_B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Check if we're running on CPU and issue warning
        if not x.is_cuda and hasattr(torch, 'has_triton') and torch.has_triton():
            print("Warning: Input tensor is on CPU, but Triton requires CUDA tensors.")
            print("Moving tensors to CUDA if available, otherwise using fallback implementation.")
            if torch.cuda.is_available():
                x = x.cuda()
                self.kn_factor_A = self.kn_factor_A.cuda()
                self.kn_factor_B = self.kn_factor_B.cuda()
                self.bias = self.bias.cuda() if self.bias is not None else None
        
        return TritonKroneckerLinearFunction.apply(x, self.kn_factor_A, self.kn_factor_B, self.bias)

    @staticmethod
    def benchmark_kronecker_linear():
        """
        Benchmark KroneckerLinear layer against standard linear layer with Kronecker product.
        Compares computation time and memory usage.
        """
        import time
        import torch
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            print("Warning: CUDA not available. Triton requires CUDA tensors. Benchmark may fail.")
            return
            
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Test dimensions
        dimensions = [
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
            (9192, 9192)
        ]
        batch_size = 256

        print("\nBenchmarking different matrix dimensions:")
        print("-" * 50)
        print(f"{'Size':>10} {'KronTime':>12} {'LinearTime':>12} {'Speedup':>10} {'MaxDiff':>10}")
        print("-" * 50)

        # Test different kron_b factors
        kron_b_factors = [2, 4, 8]

        for in_features, out_features in dimensions:
            for kron_b in kron_b_factors:
                # Create random input
                x = torch.randn(batch_size, in_features, device=device)
                
                # Method 1: Using KroneckerLinear
                # Calculate complementary factor for kron_b
                kron_a = in_features // kron_b
                
                kron_linear = KroneckerLinear(
                    in_features=in_features,
                    out_features=out_features,
                    in_factors=(kron_a, kron_b),
                    out_factors=(kron_a, kron_b),
                    bias=True,
                    device=device
                )
                
                # Method 2: Using standard linear with Kronecker product
                m1, n1 = kron_linear.out_factor1, kron_linear.in_factor1
                m2, n2 = kron_linear.out_factor2, kron_linear.in_factor2
                
                A = kron_linear.kn_factor_A.view(m1, n1)
                B = kron_linear.kn_factor_B.view(m2, n2)
                
                # Compute full weight matrix using Kronecker product
                W = torch.kron(A, B)
                linear = torch.nn.Linear(in_features, out_features, device=device)
                linear.weight.data = W
                linear.bias.data = kron_linear.bias.data

                try:
                    # Warmup runs
                    for _ in range(10):
                        y1 = kron_linear(x)
                        y2 = linear(x)
                        
                    # Benchmark KroneckerLinear with compiled forward
                    kron_linear_compiled = torch.compile(kron_linear)
                    torch.cuda.synchronize()
                    start_time = time.time()
                    for _ in range(100):
                        y1 = kron_linear(x)
                    torch.cuda.synchronize() 
                    kron_time = (time.time() - start_time) / 100

                    # Benchmark standard linear with compiled forward
                    linear_compiled = torch.compile(linear)
                    torch.cuda.synchronize()
                    start_time = time.time()
                    for _ in range(100):
                        y2 = linear(x)
                    torch.cuda.synchronize()
                    linear_time = (time.time() - start_time) / 100
                    
                    # Compare outputs
                    max_diff = torch.max(torch.abs(y1 - y2))
                    speedup = linear_time/kron_time

                    print(f"{in_features:>10} {kron_time:>12.6f} {linear_time:>12.6f} {speedup:>10.2f} {max_diff:>10.6f}")
                    print(f"Kron_b factor: {kron_b}")
                except Exception as e:
                    print(f"Error during benchmark: {e}")
                    print("Try using KroneckerLinear implementation instead of TritonKroneckerLinear")


def test_correctness():
    """Test correctness of the optimized implementation"""
    device = torch.device('cuda')
    
    # Simple test case
    M, K = 32, 64  # Input: 32x64
    A_N, A_K = 8, 16  # A: 8x16
    B_N, B_K = 4, 4   # B: 4x4
    
    # Create test matrices
    x = torch.randn(M, K, device=device)
    A = torch.randn(A_N, A_K, device=device)
    B = torch.randn(B_N, B_K, device=device)
    bias = torch.randn(A_N * B_N, device=device)
    
    # Reference computation using torch.kron
    W_ref = torch.kron(A, B)
    y_ref = x @ W_ref.t() + bias
    
    # Our optimized computation
    y_opt = efficient_kronecker_linear(x, A, B, bias)
    
    # Compare results
    max_diff = torch.max(torch.abs(y_ref - y_opt))
    print(f"Correctness test - Max difference: {max_diff:.10f}")
    print(f"Shapes - Reference: {y_ref.shape}, Optimized: {y_opt.shape}")
    print(f"All close (1e-5): {torch.allclose(y_ref, y_opt, atol=1e-5)}")
    print(f"All close (1e-6): {torch.allclose(y_ref, y_opt, atol=1e-6)}")
    
    return max_diff < 1e-5


if __name__ == "__main__":
    print("Testing correctness...")
    if test_correctness():
        print("✓ Correctness test passed!")
        print("\nRunning benchmark...")
        KroneckerLinear.benchmark_kronecker_linear()
         # Set up input sizes
        M, K = 2, 4
        A_N, A_K = 2, 2
        B_N, B_K = 2, 2

        # Create double precision inputs
        device = torch.device('cuda')
        x = torch.randn(M, K, device=device, requires_grad=True)
        A = torch.randn(A_N, A_K, device=device, requires_grad=True)
        B = torch.randn(B_N, B_K, device=device, requires_grad=True)
        bias = torch.randn(A_N * B_N, device=device, requires_grad=True)
        # Forward with your custom function
        y1 = TritonKroneckerLinearFunction.apply(x, A, B, bias)
        breakpoint()
        loss1 = y1.sum()
        loss1.backward()
        grad_x1 = x.grad.clone()
        grad_A1 = A.grad.clone()
        grad_B1 = B.grad.clone()
        grad_bias1 = bias.grad.clone()

        # Zero gradients
        x.grad.zero_(); A.grad.zero_(); B.grad.zero_(); bias.grad.zero_()

        # Forward with reference (e.g., using torch.kron)
        W = torch.kron(A, B)
        y2 = x @ W.t() + bias
        breakpoint()
        loss2 = y2.sum()
        loss2.backward()
        grad_x2 = x.grad.clone()
        grad_A2 = A.grad.clone()
        grad_B2 = B.grad.clone()
        grad_bias2 = bias.grad.clone()

        # Compare
        print(torch.allclose(grad_x1, grad_x2, atol=1e-5))
        print(torch.allclose(grad_A1, grad_A2, atol=1e-5))
        print(torch.allclose(grad_B1, grad_B2, atol=1e-5))
        print(torch.allclose(grad_bias1, grad_bias2, atol=1e-5))

    else:
        print("✗ Correctness test failed!") 