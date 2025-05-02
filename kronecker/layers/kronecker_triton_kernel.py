import triton
import triton.language as tl
import torch
import math
from typing import Optional, Tuple
import torch.nn as nn
# Default tile sizes for grid computation
DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 64

# Autotuning configurations
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def linear_kernel(
    x_ptr, a_kron_factor_ptr, b_kron_factor_ptr, bias_ptr, out_ptr,
    M, K, N, b_Kron_factor_N, b_Kron_factor_K, 
    sxm, sxk,
    sak, san, # strides for a_kron_factor.transpose
    sbk,sbn,  # strides for b_kron_factor.transpose
    sb, # stride for bias
    som, son,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):



    #import pdb; pdb.set_trace()
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    b_kron_factor_BLOCK_N = b_Kron_factor_N
    b_kron_factor_BLOCK_K = b_Kron_factor_K
    a_kron_factor_BLOCK_N = BLOCK_N//b_kron_factor_BLOCK_N
    a_kron_factor_BLOCK_K = BLOCK_K//b_kron_factor_BLOCK_K

    a_kron_factor_N = N//b_Kron_factor_N
    a_kron_factor_K = K//b_Kron_factor_K

    offs_b_kron_factor_n = pid_n + tl.arange(0, b_kron_factor_BLOCK_N)
    offs_a_kron_factor_n = pid_n + tl.arange(0, a_kron_factor_BLOCK_N)
    offs_b_kron_factor_k = tl.arange(0, b_kron_factor_BLOCK_K)
   
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        offs_a_kron_factor_k = tl.arange(0, a_kron_factor_BLOCK_K)

        x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        
        b_kron_factor = tl.load(b_kron_factor_ptr + offs_b_kron_factor_n[:, None] * sbn + offs_b_kron_factor_k[None, :] * sbk,
                                mask=(offs_b_kron_factor_n[None, :] < b_Kron_factor_N) & (offs_b_kron_factor_k[:, None] < b_Kron_factor_K), other=0.0)
        

        a_kron_factor = tl.load(a_kron_factor_ptr + offs_a_kron_factor_n[:, None] * san + offs_a_kron_factor_k[None, :] * sak,
                                mask=(offs_a_kron_factor_n[None, :] < a_kron_factor_N) & (offs_a_kron_factor_k[:, None] < a_kron_factor_K), other=0.0)
        
        # Initialize accumulator for Kronecker product
        kron_result = tl.zeros((a_kron_factor_BLOCK_N * b_kron_factor_BLOCK_N, a_kron_factor_BLOCK_K * b_kron_factor_BLOCK_K), dtype=tl.float32)
        
        # Compute Kronecker product
        for i in range(a_kron_factor_BLOCK_N):
            for j in range(a_kron_factor_BLOCK_K):
                for p in range(b_kron_factor_BLOCK_N):
                    for q in range(b_kron_factor_BLOCK_K):
                        row_idx = i * b_kron_factor_BLOCK_N + p
                        col_idx = j * b_kron_factor_BLOCK_K + q
                        kron_result[row_idx, col_idx] = a_kron_factor[i, j] * b_kron_factor[p, q]

        
        #w = tl.load(w_ptr + offs_n[None, :] * swn + offs_k[:, None] * swk,
        #            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(x, kron_result)


    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]
    tl.store(out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))



def efficient_kronecker_linear(x, a_kron_factor,b_kron_factor, bias):
    """Launch Triton linear kernel"""
    M, K = x.shape
    N = a_kron_factor.shape[1] * b_kron_factor.shape[1]
    b_Kron_factor_N = b_kron_factor.shape[0] 
    b_Kron_factor_K = b_kron_factor.shape[1]  
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, DEFAULT_BLOCK_M), triton.cdiv(N, DEFAULT_BLOCK_N))
    linear_kernel[grid](
        x, a_kron_factor, b_kron_factor, bias, out,
        M, K, N, b_Kron_factor_N, b_Kron_factor_K,
        x.stride(0), x.stride(1),
        a_kron_factor.stride(1), a_kron_factor.stride(0),
        b_kron_factor.stride(1), b_kron_factor.stride(0),
        bias.stride(0),
        out.stride(0), out.stride(1)
    )
    return out


class TritonKroneckerLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_kron_factor, b_kron_factor, bias):
        ctx.save_for_backward(x, a_kron_factor, b_kron_factor)
        return efficient_kronecker_linear(x, a_kron_factor, b_kron_factor, bias)
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Backward for TritonKroneckerLinear not yet implemented")
    # @staticmethod
    # def backward(ctx, grad_out):
    #     x, a_kron_factor, b_kron_factor = ctx.saved_tensors
    #     grad_bias = grad_out.sum(dim=0)
    #     grad_a_kron_factor = torch.matmul(grad_out.t(), x)
    #     grad_b_kron_factor = torch.matmul(grad_out, a_kron_factor)
    #     return grad_x, grad_weight, grad_bias


class TritonKroneckerLinear(torch.nn.Module):
        
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
        return TritonKroneckerLinearFunction.apply(x, self.kn_factor_A, self.kn_factor_B , self.bias)



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
        in_features = 100
        out_features = 100
        batch_size = 32
        
        # Create random input
        x = torch.randn(batch_size, in_features)
        
        # Method 1: Using KroneckerLinear
        kron_linear = TritonKroneckerLinear(
            in_features=in_features,
            out_features=out_features,
            in_factors=(50, 2),
            out_factors=(50, 2),
            bias=True
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
    TritonKroneckerLinear.benchmark_kronecker_linear()

if __name__ == "__main__":
    __main__()

