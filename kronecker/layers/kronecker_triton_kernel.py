import triton
import triton.language as tl
import torch
import math
from typing import Optional, Tuple
import torch.nn as nn
from typing import Optional, Tuple
import torch.nn as nn
# Default tile sizes for grid computation
import os
#os.environ["TRITON_INTERPRET"]="1"
# Autotuning configurations
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_M': 512, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=8),
        # triton.Config({'BLOCK_M': 512, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=5, num_warps=8)
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_M': 512, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=8),
        # triton.Config({'BLOCK_M': 512, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=5, num_warps=8)
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def linear_kernel(
    x_ptr, a_kron_factor_ptr, b_kron_factor_ptr, bias_ptr, out_ptr,
    M, K, N, 
    B_KRON_FACTOR_BLOCK_N:tl.constexpr, B_KRON_FACTOR_BLOCK_K:tl.constexpr, #here we assume that B is one block
    A_KRON_FACTOR_N:tl.constexpr, A_KRON_FACTOR_K:tl.constexpr,
    #A_KRON_FACTOR_BLOCK_N:tl.constexpr, A_KRON_FACTOR_BLOCK_K:tl.constexpr,
    x_ptr, a_kron_factor_ptr, b_kron_factor_ptr, bias_ptr, out_ptr,
    M, K, N, 
    B_KRON_FACTOR_BLOCK_N:tl.constexpr, B_KRON_FACTOR_BLOCK_K:tl.constexpr, #here we assume that B is one block
    A_KRON_FACTOR_N:tl.constexpr, A_KRON_FACTOR_K:tl.constexpr,
    #A_KRON_FACTOR_BLOCK_N:tl.constexpr, A_KRON_FACTOR_BLOCK_K:tl.constexpr,
    sxm, sxk,
    sak, san, # strides for a_kron_factor.transpose
    sbk,sbn,  # strides for b_kron_factor.transpose
    sb, # stride for bias
    sak, san, # strides for a_kron_factor.transpose
    sbk,sbn,  # strides for b_kron_factor.transpose
    sb, # stride for bias
    som, son,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):



   
  
    A_KRON_FACTOR_BLOCK_N: tl.constexpr = BLOCK_N // B_KRON_FACTOR_BLOCK_N
    A_KRON_FACTOR_BLOCK_K: tl.constexpr = BLOCK_K // B_KRON_FACTOR_BLOCK_K

    # a_kron_factor_N = N//b_Kron_factor_N
    # a_kron_factor_K = K//b_Kron_factor_K


    #import pdb; pdb.set_trace()



   
  
    A_KRON_FACTOR_BLOCK_N: tl.constexpr = BLOCK_N // B_KRON_FACTOR_BLOCK_N
    A_KRON_FACTOR_BLOCK_K: tl.constexpr = BLOCK_K // B_KRON_FACTOR_BLOCK_K

    # a_kron_factor_N = N//b_Kron_factor_N
    # a_kron_factor_K = K//b_Kron_factor_K


    #import pdb; pdb.set_trace()
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)


    
    offs_a_kron_factor_n = pid_n * A_KRON_FACTOR_BLOCK_N + tl.arange(0, A_KRON_FACTOR_BLOCK_N)
    offs_b_kron_factor_n =  tl.arange(0, B_KRON_FACTOR_BLOCK_N) #pid_n +
    offs_b_kron_factor_k = tl.arange(0, B_KRON_FACTOR_BLOCK_K)

    
    offs_a_kron_factor_n = pid_n * A_KRON_FACTOR_BLOCK_N + tl.arange(0, A_KRON_FACTOR_BLOCK_N)
    offs_b_kron_factor_n =  tl.arange(0, B_KRON_FACTOR_BLOCK_N) #pid_n +
    offs_b_kron_factor_k = tl.arange(0, B_KRON_FACTOR_BLOCK_K)
   
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        base_col = (k // BLOCK_K) * A_KRON_FACTOR_BLOCK_K
        offs_a_kron_factor_k = base_col + tl.arange(0, A_KRON_FACTOR_BLOCK_K)
        
        x_blk = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        
        b_blk = tl.load(b_kron_factor_ptr + offs_b_kron_factor_n[:, None] * sbn + offs_b_kron_factor_k[None, :] * sbk,
                                mask=(offs_b_kron_factor_n[:, None] < B_KRON_FACTOR_BLOCK_N) & (offs_b_kron_factor_k[None, :] < B_KRON_FACTOR_BLOCK_K), other=0.0, cache_modifier='.ca')
        b_blk = tl.load(b_kron_factor_ptr + offs_b_kron_factor_n[:, None] * sbn + offs_b_kron_factor_k[None, :] * sbk,
                                mask=(offs_b_kron_factor_n[:, None] < B_KRON_FACTOR_BLOCK_N) & (offs_b_kron_factor_k[None, :] < B_KRON_FACTOR_BLOCK_K), other=0.0, cache_modifier='.ca')
        

        a_blk = tl.load(a_kron_factor_ptr + offs_a_kron_factor_n[:, None] * san + offs_a_kron_factor_k[None, :] * sak,
                                mask=(offs_a_kron_factor_n[:, None] < A_KRON_FACTOR_N) & (offs_a_kron_factor_k[None, :] < A_KRON_FACTOR_K), other=0.0, cache_modifier='.ca')
        
                
        # # 1) fold K into (M·A_Kb, B_Kb)
        # x1 = tl.reshape(
        #     x_blk,
        #     (BLOCK_M * A_KRON_FACTOR_BLOCK_K,
        #     B_KRON_FACTOR_BLOCK_K)
        # )

        # # 2) (M·A_Kb, B_Kb) × (B_Kb, B_Nb) → (M·A_Kb, B_Nb)
        # b_blk_t = tl.trans(b_blk, (1, 0))    # swap dim0↔dim1

        # tmp = tl.dot(x1, b_blk_t)

        # # 3) reshape & permute to (M, B_Nb, A_Kb)
        # tmp2 = tl.reshape(tmp, (BLOCK_M, A_KRON_FACTOR_BLOCK_K, B_KRON_FACTOR_BLOCK_N))
        # tmp3 = tl.trans(tmp2, (0, 2, 1))

        # # 4) fold into (M·B_Nb, A_Kb)
        # tmp4 = tl.reshape(tmp3, (BLOCK_M * B_KRON_FACTOR_BLOCK_N,
        #                         A_KRON_FACTOR_BLOCK_K))

        # # 5) (M·B_Nb, A_Kb) × (A_Kb, A_Nb) → (M·B_Nb, A_Nb)
        # out4 = tl.dot(tmp4, tl.trans(a_blk))

        # # 6) back to (M, N) and accumulate
        # out_blk = tl.reshape(out4,
        #     (BLOCK_M,
        #     B_KRON_FACTOR_BLOCK_N * A_KRON_FACTOR_BLOCK_N)
        # )
        # acc += out_blk


        # after loading a_blk and b_blk with corrected masks:
        # a_blk: (ANb, AKb)
        # b_blk: (BNb, BKb)

        # 1) lift dims & broadcast
        a4 = tl.reshape(a_blk, (A_KRON_FACTOR_BLOCK_N, 1,
                                A_KRON_FACTOR_BLOCK_K, 1))
        b4 = tl.reshape(b_blk, (1, B_KRON_FACTOR_BLOCK_N,
                                1, B_KRON_FACTOR_BLOCK_K))
        k4 = a4 * b4  # → (ANb,BNb,AKb,BKb)

        # permute to (AKb, BKb, ANb, BNb)
        k4_t = tl.trans(k4, (2, 3, 0, 1))
        # now flatten into (AKb*BKb, ANb*BNb)
        kron_mat = tl.reshape(
            k4_t,
            (A_KRON_FACTOR_BLOCK_K * B_KRON_FACTOR_BLOCK_K,
             A_KRON_FACTOR_BLOCK_N * B_KRON_FACTOR_BLOCK_N)
        )
        # dot: x is (BLOCK_M, BLOCK_K), kron_mat is (BLOCK_K, BLOCK_N)
        acc += tl.dot(x_blk, kron_mat)
       

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]
    tl.store(out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))



def efficient_kronecker_linear(x, a_kron_factor,b_kron_factor, bias):
def efficient_kronecker_linear(x, a_kron_factor,b_kron_factor, bias):
    """Launch Triton linear kernel"""
    M, K = x.shape
    N = a_kron_factor.shape[1] * b_kron_factor.shape[1]
    b_kron_factor_N = b_kron_factor.shape[0] 
    b_kron_factor_K = b_kron_factor.shape[1]  

    a_kron_factor_N = a_kron_factor.shape[0]
    a_kron_factor_K = a_kron_factor.shape[1]

    N = a_kron_factor.shape[1] * b_kron_factor.shape[1]
    b_kron_factor_N = b_kron_factor.shape[0] 
    b_kron_factor_K = b_kron_factor.shape[1]  

    a_kron_factor_N = a_kron_factor.shape[0]
    a_kron_factor_K = a_kron_factor.shape[1]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, a_kron_factor, b_kron_factor, bias, out,
        x, a_kron_factor, b_kron_factor, bias, out,
        M, K, N,
        b_kron_factor_N, b_kron_factor_K,
        a_kron_factor_N, a_kron_factor_K,
        b_kron_factor_N, b_kron_factor_K,
        a_kron_factor_N, a_kron_factor_K,
        x.stride(0), x.stride(1),
        a_kron_factor.stride(1), a_kron_factor.stride(0),
        b_kron_factor.stride(1), b_kron_factor.stride(0),
        a_kron_factor.stride(1), a_kron_factor.stride(0),
        b_kron_factor.stride(1), b_kron_factor.stride(0),
        bias.stride(0),
        out.stride(0), out.stride(1)
        out.stride(0), out.stride(1)
    )
    return out


class TritonKroneckerLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_kron_factor, b_kron_factor, bias):
        ctx.save_for_backward(x, a_kron_factor, b_kron_factor)
        return efficient_kronecker_linear(x, a_kron_factor, b_kron_factor, bias)
    
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
        """Initialize the Kronecker factors using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.kn_factor_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.kn_factor_B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
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
                
                kron_linear = TritonKroneckerLinear(
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

def __main__():
    TritonKroneckerLinear.benchmark_kronecker_linear()

if __name__ == "__main__":
    __main__()


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
                
                kron_linear = TritonKroneckerLinear(
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

def __main__():
    TritonKroneckerLinear.benchmark_kronecker_linear()

if __name__ == "__main__":
    __main__()


