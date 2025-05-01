import torch
import triton
import triton.language as tl

# --- Triton Kernel for standard linear ---
@triton.jit
def linear_tile_kernel(
    ptr_input, ptr_weight, ptr_bias, ptr_output,
    B, K, N,
    stride_in_row, stride_wt_row, stride_out_row,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_in_col=1, stride_wt_col=1, stride_out_col=1,
):
    """
    Compute a [BLOCK_M x BLOCK_N] tile of output = input @ weight^T + bias
    input: [B, K], weight: [N, K]
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    row_start = (pid // num_pid_n) * BLOCK_M
    col_start = (pid % num_pid_n) * BLOCK_N

    # Pointers
    ptr_in = ptr_input + row_start * stride_in_row
    ptr_wt = ptr_weight + col_start * stride_wt_row
    ptr_out = ptr_output + row_start * stride_out_row + col_start * stride_out_col

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tiled multiply
    for k in range(0, K, BLOCK_K):
        offs_in = ptr_in + tl.arange(0, BLOCK_M)[:, None] * stride_in_row + (k + tl.arange(0, BLOCK_K)[None, :]) * stride_in_col
        mask_in = (row_start + tl.arange(0, BLOCK_M)[:, None] < B) & (k + tl.arange(0, BLOCK_K)[None, :] < K)
        in_block = tl.load(offs_in, mask=mask_in, other=0.0)

        offs_wt = ptr_wt + (k * stride_wt_col) + tl.arange(0, BLOCK_K)[:, None] * stride_wt_row + tl.arange(0, BLOCK_N)[None, :] * stride_wt_col
        mask_wt = (col_start + tl.arange(0, BLOCK_N)[None, :] < N) & (k + tl.arange(0, BLOCK_K)[:, None] < K)
        wt_block = tl.load(offs_wt, mask=mask_wt, other=0.0)

        acc += tl.dot(in_block, wt_block)

    if ptr_bias != 0:
        bias_offs = ptr_bias + col_start + tl.arange(0, BLOCK_N)[None, :]
        mask_bias = col_start + tl.arange(0, BLOCK_N)[None, :] < N
        bias_vec = tl.load(bias_offs, mask=mask_bias, other=0.0)
        acc += bias_vec[None, :]

    offs_out = ptr_out + tl.arange(0, BLOCK_M)[:, None] * stride_out_row + tl.arange(0, BLOCK_N)[None, :] * stride_out_col
    mask_out = (row_start + tl.arange(0, BLOCK_M)[:, None] < B) & (col_start + tl.arange(0, BLOCK_N)[None, :] < N)
    tl.store(offs_out, acc, mask=mask_out)

# --- Autograd Function & Wrappers ---
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['B', 'K', 'N'],
)
@triton.jit
def linear_tile_kernel_autotune(
    ptr_input, ptr_weight, ptr_bias, ptr_output,
    B, K, N,
    stride_in_row, stride_wt_row, stride_out_row,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_in_col=1, stride_wt_col=1, stride_out_col=1,
):
    """
    Autotuned version of the linear kernel
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    row_start = (pid // num_pid_n) * BLOCK_M
    col_start = (pid % num_pid_n) * BLOCK_N

    # Pointers
    ptr_in = ptr_input + row_start * stride_in_row
    ptr_wt = ptr_weight + col_start * stride_wt_row
    ptr_out = ptr_output + row_start * stride_out_row + col_start * stride_out_col

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tiled multiply
    for k in range(0, K, BLOCK_K):
        offs_in = ptr_in + tl.arange(0, BLOCK_M)[:, None] * stride_in_row + (k + tl.arange(0, BLOCK_K)[None, :]) * stride_in_col
        mask_in = (row_start + tl.arange(0, BLOCK_M)[:, None] < B) & (k + tl.arange(0, BLOCK_K)[None, :] < K)
        in_block = tl.load(offs_in, mask=mask_in, other=0.0)

        offs_wt = ptr_wt + (k * stride_wt_col) + tl.arange(0, BLOCK_K)[:, None] * stride_wt_row + tl.arange(0, BLOCK_N)[None, :] * stride_wt_col
        mask_wt = (col_start + tl.arange(0, BLOCK_N)[None, :] < N) & (k + tl.arange(0, BLOCK_K)[:, None] < K)
        wt_block = tl.load(offs_wt, mask=mask_wt, other=0.0)

        acc += tl.dot(in_block, wt_block)

    if ptr_bias != 0:
        bias_offs = ptr_bias + col_start + tl.arange(0, BLOCK_N)[None, :]
        mask_bias = col_start + tl.arange(0, BLOCK_N)[None, :] < N
        bias_vec = tl.load(bias_offs, mask=mask_bias, other=0.0)
        acc += bias_vec[None, :]

    offs_out = ptr_out + tl.arange(0, BLOCK_M)[:, None] * stride_out_row + tl.arange(0, BLOCK_N)[None, :] * stride_out_col
    mask_out = (row_start + tl.arange(0, BLOCK_M)[:, None] < B) & (col_start + tl.arange(0, BLOCK_N)[None, :] < N)
    tl.store(offs_out, acc, mask=mask_out)

def triton_linear_forward(input: torch.Tensor,
                          weight: torch.Tensor,
                          bias: torch.Tensor = None) -> torch.Tensor:
    B, K = input.shape
    N, K_w = weight.shape
    assert K == K_w, f"K mismatch: {K} vs {K_w}"
    output = torch.empty((B, N), device=input.device, dtype=input.dtype)
    grid = (triton.cdiv(B, 128) * triton.cdiv(N, 64),)  # Default grid size
    bias_ptr = bias.data_ptr() if bias is not None else 0
    linear_tile_kernel_autotune[grid](
        input.data_ptr(), weight.data_ptr(), bias_ptr, output.data_ptr(),
        B, K, N,
        input.stride(0), weight.stride(0), output.stride(0),
        input.stride(1), weight.stride(1), output.stride(1)
    )
    return output

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        y = triton_linear_forward(input, weight, bias)
        ctx.save_for_backward(input, weight, bias)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)
        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None

def triton_linear(input: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor = None) -> torch.Tensor:
    return TritonLinearFunction.apply(input, weight, bias)

# --- Kronecker Linear via two Triton calls ---
def triton_kron_linear(input: torch.Tensor,
                       A: torch.Tensor,
                       B: torch.Tensor,
                       bias: torch.Tensor = None) -> torch.Tensor:
    """
    Linear layer where weight = kron(A, B).
    Args:
      input: [batch, IA*IB]
      A: [OA, IA], B: [OB, IB]
      bias: [OA*OB] or None
    Returns:
      output: [batch, OA*OB]
    """
    batch, dim_in = input.shape
    IA = A.shape[1]; IB = B.shape[1]
    OA = A.shape[0]; OB = B.shape[0]
    assert dim_in == IA * IB, "Input dim mismatch"
    # Step 1: apply B -> reshape input to [batch*IA, IB]
    x = input.view(batch, IA, IB).reshape(batch * IA, IB)
    y1 = triton_linear_forward(x, B)  # [batch*IA, OB]
    # Step 2: apply A -> reshape to [batch, IA, OB] -> permute to [batch*OB, IA]
    y1 = y1.view(batch, IA, OB).permute(0, 2, 1).reshape(batch * OB, IA)
    y2 = triton_linear_forward(y1, A)  # [batch*OB, OA]
    # Reshape back and flatten
    y2 = y2.view(batch, OB, OA).permute(0, 2, 1).reshape(batch, OA * OB)
    if bias is not None:
        y2 = y2 + bias
    return y2

# --- Tests ---
if __name__ == "__main__":
    torch.manual_seed(0)
    for dtype in [torch.float32, torch.float16]:
        # standard linear test
        B, K, N = 32, 64, 128
        x = torch.randn(B, K, device='cuda', dtype=dtype)
        W = torch.randn(N, K, device='cuda', dtype=dtype)
        b = torch.randn(N, device='cuda', dtype=dtype)
        y_ref = torch.nn.functional.linear(x, W, b)
        y_tr = triton_linear(x, W, b)
        err = (y_ref - y_tr).abs().max()
        print(f"linear {dtype}: max err={err.item()}")
        assert torch.allclose(y_ref, y_tr, atol=1e-2, rtol=1e-2)
        
        # kron linear test
        IA, IB, OA, OB = 8, 16, 12, 10
        A = torch.randn(OA, IA, device='cuda', dtype=dtype)
        Bf = torch.randn(OB, IB, device='cuda', dtype=dtype)
        bias_k = torch.randn(OA*OB, device='cuda', dtype=dtype)
        xk = torch.randn(B, IA*IB, device='cuda', dtype=dtype)
        Wk = torch.kron(A, Bf).to(dtype)
        yk_ref = torch.nn.functional.linear(xk, Wk, bias_k)
        yk_tr = triton_kron_linear(xk, A, Bf, bias_k)
        errk = (yk_ref - yk_tr).abs().max()
        print(f"kron_linear {dtype}: max err={errk.item()}")
        assert torch.allclose(yk_ref, yk_tr, atol=1e-2, rtol=1e-2)
    print("All tests passed.")
