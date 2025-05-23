import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class KroneckerLinear(nn.Module):
    """
    An efficient Kronecker product-based linear layer that decomposes a large weight matrix
    into a sum of Kronecker products of smaller matrices to reduce the number of parameters.
    
    The weight matrix is approximated as:
      W ≈ sum_{i=1}^{num_sum} (A_i ⊗ B_i)
    
    This reduces the parameters from out_features * in_features to 
      num_sum * (out_factor1 * in_factor1 + out_factor2 * in_factor2)
    
    Additional functionality:
      - Efficient computation that avoids forming full matrix.
      - Initialization from a standard torch.nn.Linear through SVD-based nearest Kronecker product decomposition.
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
        num_sum: int = 1,
        efficient_sum: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_sum = num_sum
        self.efficient_sum = efficient_sum

        # Determine factors if not provided
        if in_factors is None:
            in_factors = self._get_optimal_factors(in_features)
        if out_factors is None:
            out_factors = self._get_optimal_factors(out_features)
        
        # Validate that factors multiply to the correct dimensions
        if in_factors[0] * in_factors[1] != in_features:
            raise ValueError(f"in_factors {in_factors} must multiply to in_features {in_features}")
        if out_factors[0] * out_factors[1] != out_features:
            raise ValueError(f"out_factors {out_factors} must multiply to out_features {out_features}")
        
        # Descriptive factor naming
        self.in_factor1, self.in_factor2 = in_factors
        self.out_factor1, self.out_factor2 = out_factors

        # Pre-compute an efficiency flag based on the factors for the order of multiplications
        self.use_bXtA_path = ((2 * self.in_factor2 - 1) * self.out_factor2 * self.in_factor1 +
                              (2 * self.in_factor1 - 1) * self.out_factor2 * self.out_factor1) < (
                              (2 * self.in_factor1 - 1) * self.in_factor2 * self.out_factor1 +
                              (2 * self.in_factor2 - 1) * self.out_factor2 * self.out_factor1)

        # Initialize Kronecker factors; shapes: (num_sum, out_factor, in_factor)
        self.kn_factor_A = nn.Parameter(torch.empty((num_sum, self.out_factor1, self.in_factor1), device=device, dtype=dtype))
        self.kn_factor_B = nn.Parameter(torch.empty((num_sum, self.out_factor2, self.in_factor2), device=device, dtype=dtype))
        
        # Bias parameter
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        
        self.reset_parameters()
    
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
    
    @property
    def weight(self) -> torch.Tensor:
        """
        Compute the full weight matrix as the sum of Kronecker products.
        Note: This is computationally expensive and should only be used for debugging.
        """
        return torch.sum(self.batch_kron(self.kn_factor_A, self.kn_factor_B), dim=0)
    
    @staticmethod
    def batch_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the Kronecker product for each pair in two batches."""
        return torch.stack([torch.kron(a_i, b_i) for a_i, b_i in zip(a, b)])
  
    def parallel_sum_kron(self, input: torch.Tensor) -> torch.Tensor:
            # Record if input was originally 2D.
            was_2d = (input.dim() == 2)
            if was_2d:
                input = input.unsqueeze(1)  # Now shape: (B, 1, in_features)

            B, T, _ = input.shape
            # Reshape input to (B*T, in_factor2, in_factor1)
            X = input.view(B * T, self.in_factor1, self.in_factor2).transpose(1, 2)
            
            # Estimate FLOP costs for two multiplication orderings.
            # Option 1: Multiply kn_factor_B first then kn_factor_A^T.
            cost_option1 = self.in_factor1 * self.out_factor2 * (self.in_factor2 + self.out_factor1)
            # Option 2: Multiply kn_factor_A first then kn_factor_B.
            cost_option2 = self.in_factor2 * self.out_factor1 * (self.in_factor1 + self.out_factor2)
            
            if cost_option1 <= cost_option2:
                # Option 1: Multiply with kn_factor_B first.
                # First multiplication: (B*T, s, out_factor2, in_factor1)
                temp = torch.einsum('s o i, b i j -> b s o j', 
                                    self.kn_factor_B,  # shape: (s, out_factor2, in_factor2)
                                    X)                 # shape: (B*T, in_factor2, in_factor1)
                # Second multiplication: (B*T, s, out_factor1, out_factor2)
                out_all = torch.einsum('b s o j, s j k -> b s k o', 
                                    temp,                # shape: (B*T, s, out_factor2, in_factor1)
                                    self.kn_factor_A.transpose(1, 2))  # shape: (s, in_factor1, out_factor1)
            else:
                # Option 2: Multiply with kn_factor_A first.
                # First multiplication: (B*T, s, in_factor2, out_factor1)
                temp = torch.einsum('b i j, s k j -> b s i k', 
                                    X,                  # shape: (B*T, in_factor2, in_factor1)
                                    self.kn_factor_A)   # shape: (s, out_factor1, in_factor1)
                # Second multiplication: (B*T, s, out_factor1, out_factor2)
                out_all = torch.einsum('s o i, b s i k -> b s k o', 
                                    self.kn_factor_B,  # shape: (s, out_factor2, in_factor2)
                                    temp)              # shape: (B*T, s, in_factor2, out_factor1)
            
            # Sum over the summation dimension: (B*T, out_factor1, out_factor2)
            out_sum = out_all.sum(dim=1)
            
            # Rearrange the output to (B, T, out_features) by flattening the last two dimensions.
            result = out_sum.reshape(B, T, -1)
            if was_2d:
                result = result.squeeze(1)
            return result


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        - Uses an efficient summation strategy if efficient_sum is True.
        - Otherwise, computes the full weight matrix explicitly.
        """
        if self.efficient_sum:
            x = self.parallel_sum_kron(input)
            return x + self.bias if self.bias is not None else x
        else:
            weight = torch.sum(self.batch_kron(self.kn_factor_A, self.kn_factor_B), dim=0)
            return F.linear(input, weight, self.bias)
    
    @staticmethod
    def _svd_nearest_kron(weight: torch.Tensor, m1: int, n1: int, m2: int, n2: int, num_sum: int):
        """
        Compute the nearest Kronecker product decomposition via SVD.
        Given weight of shape (m1*m2, n1*n2), rearrange it to a matrix of shape (m1*n1, m2*n2)
        and perform SVD. Use the top singular components to form the decomposition.
        
        Returns:
            Tuple (A_factors, B_factors):
              - A_factors: Tensor of shape (num_sum, m1, n1)
              - B_factors: Tensor of shape (num_sum, m2, n2)
        """
        # Rearrange weight from (m1*m2, n1*n2) to (m1*n1, m2*n2)
        R = weight.view(m1, m2, n1, n2).permute(0, 2, 1, 3).contiguous().view(m1 * n1, m2 * n2)
        # Convert to float32 for SVD, then back to original dtype
        orig_dtype = R.dtype
        R_f32 = R.to(torch.float32)
        U, S, Vh = torch.linalg.svd(R_f32, full_matrices=False)
        U, S, Vh = U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)
        components = min(num_sum, S.size(0))
        A_factors = []
        B_factors = []
        for i in range(components):
            scale = math.sqrt(S[i].item())
            A_i = scale * U[:, i].view(m1, n1)
            B_i = scale * Vh[i, :].view(m2, n2)
            A_factors.append(A_i)
            B_factors.append(B_i)
        if components < num_sum:
            extra = num_sum - components
            A_extra = torch.empty((extra, m1, n1), dtype=weight.dtype, device=weight.device)
            B_extra = torch.empty((extra, m2, n2), dtype=weight.dtype, device=weight.device)
            nn.init.kaiming_uniform_(A_extra, a=math.sqrt(5))
            nn.init.kaiming_uniform_(B_extra, a=math.sqrt(5))
            A_factors.append(A_extra)
            B_factors.append(B_extra)
            A_factors = torch.cat(A_factors, dim=0)
            B_factors = torch.cat(B_factors, dim=0)
        else:
            A_factors = torch.stack(A_factors, dim=0)
            B_factors = torch.stack(B_factors, dim=0)
        return A_factors, B_factors
    
    @classmethod
    def from_linear_with_factors(
        cls,
        linear: nn.Linear,
        out_factors: Tuple[int,int],
        in_factors: Tuple[int,int],
        num_sum: int = 1,
        efficient_sum: bool = True,
    ) -> 'KroneckerLinear':
        """
        Initialize from an existing nn.Linear, but *force* the Kronecker-factor
        shapes to out_factors (m1,m2) and in_factors (n1,n2).
        """
        m1, m2 = out_factors
        n1, n2 = in_factors
        has_bias = linear.bias is not None

        # instantiate with your chosen factors
        kron_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            in_factors=(n1,n2),
            out_factors=(m1,m2),
            bias=has_bias,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            num_sum=num_sum,
            efficient_sum=efficient_sum,
        )

        W = linear.weight.data
        kn_A, kn_B = cls._svd_nearest_kron(W, m1, n1, m2, n2, num_sum)
        kron_layer.kn_factor_A.data.copy_(kn_A)
        kron_layer.kn_factor_B.data.copy_(kn_B)
        if has_bias:
            kron_layer.bias.data.copy_(linear.bias.data)
        return kron_layer

    def extra_repr(self) -> str:
        """Return a string representation of the module for debugging."""
        original_params = self.in_features * self.out_features
        kron_params = self.kn_factor_A.numel() + self.kn_factor_B.numel() + (self.bias.numel() if self.bias is not None else 0)
        reduction = original_params / kron_params if kron_params > 0 else float('inf')
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'in_factors=({self.in_factor1}, {self.in_factor2}), out_factors=({self.out_factor1}, {self.out_factor2}), '
                f'num_sum={self.num_sum}, efficient_sum={self.efficient_sum}, '
                f'bias={self.bias is not None}, params_reduction={reduction:.2f}x')
    @staticmethod
    def benchmark_kronecker_linear():
        """
        Benchmark KroneckerLinear layer against standard linear layer with Kronecker product.
        Compares computation time and memory usage.
        """
        import time
        import torch
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Test dimensions
        in_features = 1024
        out_features = 1024
        batch_size = 32
        
        # Create random input
        x = torch.randn(batch_size, in_features)
        
        # Method 1: Using KroneckerLinear
        kron_linear = KroneckerLinear(
            in_features=in_features,
            out_features=out_features,
            num_sum=1,
            efficient_sum=True
        )
        
        # Method 2: Using standard linear with Kronecker product
        # Create random factors
        m1, n1 = kron_linear.out_factor1, kron_linear.in_factor1
        m2, n2 = kron_linear.out_factor2, kron_linear.in_factor2
        
        A = kron_linear.kn_factor_A.view(m1, n1)
        B = kron_linear.kn_factor_B.view(m2, n2)
        
        # Compute full weight matrix using Kronecker product
        W = torch.kron(A, B)
        linear = torch.nn.Linear(in_features, out_features)
        linear.weight.data = W
        
        # Benchmark KroneckerLinear
        start_time = time.time()
        for _ in range(100):
            y1 = kron_linear(x)
        kron_time = (time.time() - start_time) / 100
        
        # Benchmark standard linear
        start_time = time.time()
        for _ in range(100):
            y2 = linear(x)
        linear_time = (time.time() - start_time) / 100
        
        # Compare outputs to ensure they are acceptably close
        max_diff = torch.max(torch.abs(y1 - y2))
        print(f"Maximum difference between outputs: {max_diff:.6f}")
        print(f"KroneckerLinear time: {kron_time:.6f}s")
        print(f"Standard linear time: {linear_time:.6f}s")
        print(f"Speedup factor: {linear_time/kron_time:.2f}x")

def __main__():
    KroneckerLinear.benchmark_kronecker_linear()

if __name__ == "__main__":
    __main__()

