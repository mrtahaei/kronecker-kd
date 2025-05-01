import triton
import triton.language as tl
import torch
import math

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
    x_ptr, w_ptr, a_kron_factor_ptr, b_kron_factor_ptr, bias_ptr, out_ptr,
    M, K, N, b_Kron_factor_N, b_Kron_factor_K, 
    sxm, sxk,
    swk, swn,
    sb,
    som, son,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    import pdb; pdb.set_trace()
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



def efficient_kronecker_linear(x, weight, bias):
    """Launch Triton linear kernel"""
    M, K = x.shape
    N, _ = weight.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, DEFAULT_BLOCK_M), triton.cdiv(N, DEFAULT_BLOCK_N))
    linear_kernel[grid](
        x, weight, bias, out,
        M, K, N,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        bias.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


class TritonKroneckerLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        return efficient_kronecker_slinear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_bias = grad_out.sum(dim=0)
        grad_weight = torch.matmul(grad_out.t(), x)
        grad_x = torch.matmul(grad_out, weight)
        return grad_x, grad_weight, grad_bias


class TritonKroneckerLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return TritonKroneckerLinearFunction.apply(x, self.weight, self.bias)


# Validation and Benchmark

def validate_correctness(M=512, K=256, N=512, rtol=1e-3, atol=1e-2):
    device = 'cuda'
    x = torch.randn((M, K), device=device)
    w = torch.randn((N, K), device=device)
    b = torch.randn((N,), device=device)
    out_t = TritonLinearFunction.apply(x, w, b)
    out_p = torch.nn.functional.linear(x, w, b)
    diff = (out_t - out_p).abs()
    tol = atol + rtol * out_p.abs()
    fails = (diff > tol).sum().item()
    total = diff.numel()
    perc = fails / total * 100
    print(f"Validation: {fails}/{total} ({perc:.2f}%) elements exceed tol")


def benchmark(M=2048, K=1024, N=2048, runs=50, warmup=10):
    device = 'cuda'
    x = torch.randn((M, K), device=device)
    w = torch.randn((N, K), device=device)
    b = torch.randn((N,), device=device)

    # Prepare models
    torch_lin = torch.nn.Linear(K, N).to(device)
    torch_lin_comp = torch.compile(torch_lin)
    triton_lin = TritonLinear(K, N).to(device)
    triton_lin_comp = torch.compile(triton_lin)

    # Warmup
    for _ in range(warmup):
        _ = torch_lin(x)
        _ = torch_lin_comp(x)
        _ = TritonLinearFunction.apply(x, w, b)
        _ = triton_lin_comp(x)

    # Benchmark
    def time_fn(fn):
        torch.cuda.synchronize(); start = torch.cuda.Event(True); end = torch.cuda.Event(True)
        start.record()
        for _ in range(runs): fn(x)
        end.record(); torch.cuda.synchronize()
        return start.elapsed_time(end) / runs

    t_torch = time_fn(torch_lin)
    t_torch_comp = time_fn(torch_lin_comp)
    t_triton = time_fn(lambda inp: TritonLinearFunction.apply(inp, w, b))
    t_triton_comp = time_fn(triton_lin_comp)

    print(f"Torch         : {t_torch:.3f} ms")
    print(f"Torch Compiled: {t_torch_comp:.3f} ms")
    print(f"Triton        : {t_triton:.3f} ms")
    print(f"Triton Compiled: {t_triton_comp:.3f} ms")


if __name__ == '__main__':
    print("Running validation...")
    validate_correctness()
    print("\nRunning benchmark...")
    benchmark()
