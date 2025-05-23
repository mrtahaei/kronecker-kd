import triton
import triton.language as tl
import torch
import math
from typing import Tuple, List, Optional
import torch.nn as nn

############################################
#  Triton Kronecker Linear – B‑factor tiny  #
############################################
#  Assumption:  B_K ≤ 16  (fits in registers/L1), so we can
#  cache **entire B** for each kernel and reuse across K‑loop.
############################################

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
    ],
    key=["M","N","K"]
)
@triton.jit
def kron_linear_kernel(
    x_ptr, a_ptr, b_ptr, bias_ptr, out_ptr,
    M, K, N,
    A_N: tl.constexpr, A_K: tl.constexpr,
    B_N: tl.constexpr, B_K: tl.constexpr,
    sxm, sxk,        # X strides
    sak, san,        # A strides (row‑major: (A_N, A_K))
    sbk, sbn,        # B strides (row‑major: (B_N, B_K))
    sb,              # bias stride
    som, son,        # Y strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Y = X · (A ⊗ B) + bias.  Optimised for **small B**.
    Strategy:
    • Cache the full B (B_N×B_K) in registers once per kernel (fits since B_K small).
    • Iterate over A_K blocks inside the K‑loop, broadcasting B.
    """
    tl.static_assert(B_K <= 16, "This variant targets very small B_K (≤16)")
    tl.static_assert(BLOCK_K % B_K == 0, "BLOCK_K must be multiple of B_K")

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # rows in X/Y
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # cols in Y

    # factor row/col indices for this N‑tile
    a_row_idx = offs_n // B_N   # (BN_tile,)  idx over A_N
    b_row_idx = offs_n %  B_N   # idx over B_N

    # ---- cache full B in registers (small) ----
    B_full = tl.load(b_ptr + b_row_idx[None, :] * sbn + tl.arange(0, B_K)[:, None] * sbk,
                     cache_modifier='.ca')            # (B_K , BN_tile)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over A_K in tiles of size AK_STEP (= BLOCK_K//B_K)
    AK_STEP: tl.constexpr = BLOCK_K // B_K
    for ak_offset in range(0, A_K, AK_STEP):
        # Slice A rows needed for this tile
        a_col_slice = ak_offset + tl.arange(0, AK_STEP)            # length = AK_STEP

        # load A_sub (AK_STEP × BN_tile)
        A_sub = tl.load(a_ptr + a_row_idx[None, :] * san + a_col_slice[:, None] * sak,
                        cache_modifier='.ca')        # (AK_STEP , BN_tile)

        # Build kron tile on‑the‑fly via outer‑product broadcast
        # kron_mat shape: (AK_STEP*B_K , BN_tile)
        a4 = tl.reshape(A_sub, (AK_STEP, 1, BLOCK_N))  # (AK_STEP,1,BN_tile)
        b4 = tl.reshape(B_full, (1, B_K, BLOCK_N))     # (1,BK,BN_tile)
        kron_block = tl.reshape(a4 * b4, (AK_STEP * B_K, BLOCK_N))

        # Corresponding K indices in X
        k_idx = ak_offset * B_K + tl.arange(0, AK_STEP * B_K)
        x_blk = tl.load(x_ptr + offs_m[:, None] * sxm + k_idx[None, :] * sxk,
                        mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.)

        acc += tl.dot(x_blk, kron_block)

    # bias + store
    bias_vec = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.)
    acc += bias_vec[None, :]

    tl.store(out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# ───────────────────────── Python wrapper ─────────────────────────

def kron_linear_forward(x, A, B, bias):
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





class TritonKroneckerLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_kron_factor, b_kron_factor, bias):
        ctx.save_for_backward(x, a_kron_factor, b_kron_factor)
        return kron_linear_forward(x, a_kron_factor, b_kron_factor, bias)
    
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
