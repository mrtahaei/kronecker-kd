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
        """
        Computes the sum (over the Kronecker summation indices) of (A_i ⊗ B_i) x
        without explicitly constructing the full weight matrix.
        
        Supports inputs of shape (batch_size, in_features) or (batch_size, sequence_length, in_features).
        """
        was_2d = (input.dim() == 2)
        if was_2d:
            input = input.unsqueeze(1)  # Now shape: (B, 1, in_features)
        
        B, T, _ = input.shape
        # Reshape input to leverage the factor structure
        # New shape: (B*T, in_factor2, in_factor1) after reshaping and transposing
        input_reshaped = input.view(B * T, self.in_factor1, self.in_factor2).transpose(1, 2)
        
        # Choose multiplication path based on precomputed flag for efficiency:
        if self.use_bXtA_path:
            # Compute: y = kn_factor_B @ input_reshaped @ kn_factor_A^T
            y = torch.bmm(
                self.kn_factor_B.unsqueeze(0).expand(B * T, -1, self.out_factor2, self.in_factor2)
                    .reshape(-1, self.out_factor2, self.in_factor2),
                input_reshaped
            )
            y = torch.bmm(
                y,
                self.kn_factor_A.transpose(1, 2).unsqueeze(0).expand(B * T, -1, self.in_factor1, self.out_factor1)
                    .reshape(-1, self.in_factor1, self.out_factor1)
            )
        else:
            # Alternate path: y = kn_factor_B @ (input_reshaped @ kn_factor_A^T)
            y = torch.bmm(
                input_reshaped,
                self.kn_factor_A.transpose(1, 2).unsqueeze(0).expand(B * T, -1, self.in_factor1, self.out_factor1)
                    .reshape(-1, self.in_factor1, self.out_factor1)
            )
            y = torch.bmm(
                self.kn_factor_B.unsqueeze(0).expand(B * T, -1, self.out_factor2, self.in_factor2)
                    .reshape(-1, self.out_factor2, self.in_factor2),
                y
            )
        
        # Reshape and sum contributions across num_sum dimension:
        out = torch.sum(y.view(B * T, self.num_sum, self.out_factor2, self.out_factor1), dim=1)
        result = out.transpose(1, 2).reshape(B, T, -1)
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
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
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
    def from_linear(cls, linear: nn.Linear, num_sum: int = 1, efficient_sum: bool = True) -> 'KroneckerLinear':
        """
        Initialize a KroneckerLinear layer from an existing nn.Linear layer.
        
        This method performs an SVD-based decomposition of the linear layer's weight matrix to
        find the nearest Kronecker product representation.
        
        Args:
            linear: The torch.nn.Linear layer to convert.
            num_sum: Number of summation terms (components) for the Kronecker decomposition.
            efficient_sum: Whether to use the efficient summation implementation.
        
        Returns:
            A KroneckerLinear layer initialized from the given linear layer.
        """
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = (linear.bias is not None)
        
        # Instantiate a new KroneckerLinear layer.
        kron_layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            num_sum=num_sum,
            efficient_sum=efficient_sum
        )
        # Obtain the full weight from the linear layer (shape: [out_features, in_features])
        W = linear.weight.data
        # Factor dimensions from the new layer
        m1, n1 = kron_layer.out_factor1, kron_layer.in_factor1
        m2, n2 = kron_layer.out_factor2, kron_layer.in_factor2
        # Use SVD to initialize the Kronecker factors
        kn_A, kn_B = cls._svd_nearest_kron(W, m1, n1, m2, n2, num_sum)
        kron_layer.kn_factor_A.data = kn_A
        kron_layer.kn_factor_B.data = kn_B
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
