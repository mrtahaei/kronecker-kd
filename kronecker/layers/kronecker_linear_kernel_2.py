import triton
import triton.language as tl
import torch
import math
from typing import Tuple, List
import torch.nn as nn

############################################
#  Triton Kronecker Linear – tiled factors  #
############################################
#  Loads **only the needed slices** of A & B per K / N tile
############################################

@triton.autotune(
    configs=[
        #triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
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

# ────────── Python wrapper ──────────

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

# ────────── Module & benchmark (unchanged) ──────────
class _KronFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, B, bias):
        ctx.save_for_backward(x, A, B)
        return kron_linear_forward(x, A, B, bias)
    @staticmethod
    def backward(ctx, grad_o):
        raise NotImplementedError

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

# ────────── Benchmark utilities ──────────

def _time_many(fn, x, iters: int = 50):
    """Run `fn(x)` `iters` times and return average milliseconds."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def benchmark(dimensions: List[Tuple[int, int]] = [ (4096, 4096), (8192, 8192)],
              batch_size: int = 16,
              repeats: int = 50,
              kron_b_factors: List[int] = [2, 4,8]):
    """Compare TritonKroneckerLinear with nn.Linear holding the full Kronecker weight."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        raise RuntimeError("CUDA GPU required for Triton benchmark")

    torch.manual_seed(0)
    header = f"{'Size':>8}{'KB':>6}{'TKL ms':>10}{'Linear ms':>12}{'Speed':>8}{'MaxErr':>10}"
    print(header)
    print('-' * len(header))

    for (din, dout) in dimensions:
        for kb in kron_b_factors:
            if din % kb or dout % kb:
                continue  # incompatible factor
            ka = din // kb
            layer = TritonKroneckerLinear(din, dout,
                                          in_factors=(ka, kb),
                                          out_factors=(ka, kb),
                                          device=device)
            # materialise dense weight for reference
            W_dense = torch.kron(layer.A, layer.B).contiguous()
            dense = nn.Linear(din, dout, device=device)
            dense.weight.data.copy_(W_dense)
            dense.bias.data.copy_(layer.bias.data)

            x = torch.randn(batch_size, din, device=device)
            
            # Warmup runs
            for _ in range(100):
                layer(x)
                dense(x)
            
            # Multiple measurement runs
            tkl_times = []
            lin_times = []
            for _ in range(100):  # 3 sets of measurements
                tkl_ms = _time_many(lambda t: layer(t), x, repeats)
                lin_ms = _time_many(lambda t: dense(t), x, repeats)
                tkl_times.append(tkl_ms)
                lin_times.append(lin_ms)
            
            # Take median of measurements
            tkl_ms = torch.tensor(tkl_times).median().item()
            lin_ms = torch.tensor(lin_times).median().item()
            
            err = (layer(x) - dense(x)).abs().max().item()
            print(f"{din:>8}{kb:>6}{tkl_ms:>10.2f}{lin_ms:>12.2f}{lin_ms/tkl_ms:>8.2f}{err:>10.1e}")

if __name__ == "__main__":
    benchmark()
