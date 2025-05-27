import triton
import triton.language as tl
import torch
import math
from typing import Tuple, List
import torch.nn as nn

############################################
#  Optimized Triton Kronecker Linear      #
############################################
#  Optimizations:
#  - Move B load outside K loop when small
#  - Cache entire B matrix for reuse
#  - Expanded autotuning configurations
############################################

@triton.autotune(
    configs=[
        # High throughput configs for large matrices
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        
        # Memory bandwidth optimized configs
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        
        # Small B factor optimized configs
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
    ],
    key=["M","N","K"],
    #print_best=True,
)
@triton.jit
def kron_linear_kernel_optimized(
    x_ptr, a_ptr, b_ptr, bias_ptr, out_ptr,
    M, K, N,
    A_N: tl.constexpr, A_K: tl.constexpr,
    B_N: tl.constexpr, B_K: tl.constexpr,
    sxm, sxk,
    sak, san,
    sbk, sbn,
    sb,
    som, son,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    CACHE_B: tl.constexpr,  # Whether to cache entire B matrix
):
    """Optimized Y = X · (A ⊗ B) + bias.
    When B is small (CACHE_B=True), loads entire B matrix once and reuses it.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Pre-compute factor indices for this N-tile
    a_row_idx = offs_n // B_N   # (BLOCK_N,)  in [0 .. A_N-1]
    b_row_idx = offs_n %  B_N   # (BLOCK_N,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Cache entire B matrix if it's small enough
    if CACHE_B:
        # Load entire B matrix once (B_N x B_K)
        b_full = tl.load(
            b_ptr + tl.arange(0, B_N)[:, None] * sbn + tl.arange(0, B_K)[None, :] * sbk,
            mask=(tl.arange(0, B_N)[:, None] < B_N) & (tl.arange(0, B_K)[None, :] < B_K),
            other=0.0
        )

    # Loop over K dimension tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_col_idx = offs_k // B_K
        b_col_idx = offs_k %  B_K

        # Load A sub-tile
        A_sub_T = tl.load(
            a_ptr + a_row_idx[None, :] * san + a_col_idx[:, None] * sak,
            mask=(a_row_idx[None, :] < A_N) & (a_col_idx[:, None] < A_K),
            other=0.0
        )

        # Load or extract B sub-tile
        if CACHE_B:
            # Extract from cached B matrix (very fast)
            B_sub_T = b_full[b_row_idx[None, :], b_col_idx[:, None]]
        else:
            # Load B sub-tile from memory
            B_sub_T = tl.load(
                b_ptr + b_row_idx[None, :] * sbn + b_col_idx[:, None] * sbk,
                mask=(b_row_idx[None, :] < B_N) & (b_col_idx[:, None] < B_K),
                other=0.0
            )

        # Compute Kronecker product tile
        kron_tile = A_sub_T * B_sub_T   # (BLOCK_K, BLOCK_N)

        # Load X tile
        x_blk = tl.load(
            x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )

        # Accumulate result
        acc += tl.dot(x_blk, kron_tile)

    # Add bias and store
    if bias_ptr is not None:
        bias_vec = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias_vec[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# Keep original kernel for compatibility
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=["M","N","K"],
    #print_best=True,
)
@triton.jit
def kron_linear_kernel(
    x_ptr, a_ptr, b_ptr, bias_ptr, out_ptr,
    M, K, N,
    A_N: tl.constexpr, A_K: tl.constexpr,
    B_N: tl.constexpr, B_K: tl.constexpr,
    sxm, sxk,
    sak, san,
    sbk, sbn,
    sb,
    som, son,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Y = X · (A ⊗ B) + bias.
    Each kernel instance computes (BLOCK_M × BLOCK_N) of Y.
    Only the specific rows/cols of A and B required by this tile are loaded.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)              # rows of X / Y
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)              # cols of Y

    # Pre‑compute factor indices for this N‑tile
    a_row_idx = offs_n // B_N   # (BLOCK_N,)  in [0 .. A_N-1]
    b_row_idx = offs_n %  B_N   # (BLOCK_N,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)                       # cols of X / rows of kron‑mat
        a_col_idx = offs_k // B_K                                 # (BLOCK_K,)
        b_col_idx = offs_k %  B_K

        # ---- gather required A & B sub‑tiles ----
        # A_sub_T : (BLOCK_K , BLOCK_N)
        A_sub_T = tl.load(a_ptr + a_row_idx[None, :] * san + a_col_idx[:, None] * sak)
        # B_sub_T : (BLOCK_K , BLOCK_N)
        B_sub_T = tl.load(b_ptr + b_row_idx[None, :] * sbn + b_col_idx[:, None] * sbk)

        kron_tile = A_sub_T * B_sub_T   # (BLOCK_K , BLOCK_N)

        # ---- load X tile ----
        x_blk = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        acc += tl.dot(x_blk, kron_tile)  # (BM,BK)·(BK,BN) -> (BM,BN)

    # add bias & store
    bias_vec = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias_vec[None, :]

    tl.store(out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# ────────── Adaptive Python wrapper ──────────

def kron_linear_forward_optimized(x, A, B, bias):
    """Optimized Kronecker linear forward with adaptive B caching."""
    M, K = x.shape
    A_N, A_K = A.shape
    B_N, B_K = B.shape
    N = A_N * B_N
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    # Decide whether to cache B matrix based on its size
    # Cache B if it's small enough to fit comfortably in shared memory/registers
    cache_b = B_N * B_K <= 256 and B_K <= 16  # Heuristic for small B matrices
    
    kron_linear_kernel_optimized[grid](
        x, A, B, bias, out,
        M, K, N,
        A_N, A_K, B_N, B_K,
        x.stride(0), x.stride(1),
        A.stride(1), A.stride(0),
        B.stride(1), B.stride(0),
        bias.stride(0) if bias is not None else 0,
        out.stride(0), out.stride(1),
        CACHE_B=cache_b
    )
    
    return out


def kron_linear_forward(x, A, B, bias):
    """Original Kronecker linear forward for compatibility."""
    M, K = x.shape
    A_N, A_K = A.shape
    B_N, B_K = B.shape
    N = A_N * B_N
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    kron_linear_kernel[grid](
        x, A, B, bias, out,
        M, K, N,
        A_N, A_K, B_N, B_K,
        x.stride(0), x.stride(1),
        A.stride(1), A.stride(0),
        B.stride(1), B.stride(0),
        bias.stride(0),
        out.stride(0), out.stride(1)
    )
    #print(kron_linear_kernel.best_config)
    return out

# ────────── Module implementations ──────────

class _OptimizedKronFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, B, bias):
        ctx.save_for_backward(x, A, B)
        return kron_linear_forward_optimized(x, A, B, bias)
    @staticmethod
    def backward(ctx, grad_o):
        raise NotImplementedError

class _KronFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, B, bias):
        ctx.save_for_backward(x, A, B)
        return kron_linear_forward(x, A, B, bias)
    @staticmethod
    def backward(ctx, grad_o):
        raise NotImplementedError

class OptimizedTritonKroneckerLinear(nn.Module):
    """Optimized Triton Kronecker Linear with B caching for small matrices."""
    def __init__(self, in_features:int, out_features:int, *, in_factors:Tuple[int,int], out_factors:Tuple[int,int], bias:bool=True, device=None):
        super().__init__()
        if in_factors[0]*in_factors[1] != in_features or out_factors[0]*out_factors[1] != out_features:
            raise ValueError("factor mismatch")
        self.A = nn.Parameter(torch.empty((out_factors[0], in_factors[0]), device=device))
        self.B = nn.Parameter(torch.empty((out_factors[1], in_factors[1]), device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device)) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1/math.sqrt(self.A.shape[1]*self.B.shape[1]), 1/math.sqrt(self.A.shape[1]*self.B.shape[1]))
    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("CUDA required")
        return _OptimizedKronFn.apply(x, self.A, self.B, self.bias)

class TritonKroneckerLinear(nn.Module):
    def __init__(self, in_features:int, out_features:int, *, in_factors:Tuple[int,int], out_factors:Tuple[int,int], bias:bool=True, device=None):
        super().__init__()
        if in_factors[0]*in_factors[1] != in_features or out_factors[0]*out_factors[1] != out_features:
            raise ValueError("factor mismatch")
        self.A = nn.Parameter(torch.empty((out_factors[0], in_factors[0]), device=device))
        self.B = nn.Parameter(torch.empty((out_factors[1], in_factors[1]), device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device)) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1/math.sqrt(self.A.shape[1]*self.B.shape[1]), 1/math.sqrt(self.A.shape[1]*self.B.shape[1]))
    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("CUDA required")
        return _KronFn.apply(x, self.A, self.B, self.bias)

       

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
                try:
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
                    A = kron_linear.A
                    B = kron_linear.B
                    
                    # Compute full weight matrix using Kronecker product
                    W = torch.kron(A, B)
                    linear = torch.nn.Linear(in_features, out_features, device=device)
                    linear.weight.data = W
                    linear.bias.data = kron_linear.bias.data

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

if __name__ == "__main__":
    TritonKroneckerLinear.benchmark_kronecker_linear()
