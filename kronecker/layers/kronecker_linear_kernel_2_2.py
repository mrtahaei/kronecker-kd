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
    """Y = X · (A ⊗ B) + bias using TRUE linear algebra identity with padding.
    
    Mathematical Identity: (A ⊗ B) * vec(X) = vec(B * X_reshaped * A^T)
    
    This kernel implements the ACTUAL linear algebra identity by:
    1. Padding B matrix to meet Triton's minimum size requirements (16x16)
    2. Using a simplified approach that computes the identity directly
    3. Avoiding complex indexing by using broadcasting and masking
    
    The approach computes: result[m, a*B_N + b] = sum_k(B[b,k] * X[m, a_k*B_K + k] * A[a, a_k])
    where a_k ranges over A_K, which is equivalent to the linear algebra identity.
    """
    tl.static_assert(B_K <= 16, "B_K must be small enough to pad efficiently")
    tl.static_assert(A_K <= 64, "A_K should be reasonably small")

    # Define padded sizes to meet Triton's minimum requirements
    B_N_PADDED: tl.constexpr = 16
    B_K_PADDED: tl.constexpr = 16

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # batch indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # output feature indices

    # Determine which (a_idx, b_idx) pairs this output tile covers
    # Output layout: out[m, a*B_N + b] = result from the linear algebra identity
    a_indices = offs_n // B_N  # which A output rows (0 to A_N-1)
    b_indices = offs_n % B_N   # which B output rows (0 to B_N-1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load B matrix with padding to meet Triton's size requirements
    b_offs_n = tl.arange(0, B_N_PADDED)
    b_offs_k = tl.arange(0, B_K_PADDED)
    
    # Load B with padding - use zeros for padded regions
    B_matrix_padded = tl.load(b_ptr + b_offs_n[:, None] * sbn + b_offs_k[None, :] * sbk,
                             mask=(b_offs_n[:, None] < B_N) & (b_offs_k[None, :] < B_K),
                             other=0.0, cache_modifier='.ca')  # (B_N_PADDED, B_K_PADDED)

    # Implement the linear algebra identity using direct computation
    # For each A_K slice, compute the contribution to the final result
    
    for a_k in range(A_K):
        # Load X slice: X[:, a_k*B_K:(a_k+1)*B_K] with padding
        x_slice_start = a_k * B_K
        x_slice_indices = x_slice_start + tl.arange(0, B_K_PADDED)
        
        # Load X slice with padding (zeros for padded columns)
        X_slice_padded = tl.load(x_ptr + offs_m[:, None] * sxm + x_slice_indices[None, :] * sxk,
                                mask=(offs_m[:, None] < M) & (x_slice_indices[None, :] < K),
                                other=0.0, cache_modifier='.ca')  # (BLOCK_M, B_K_PADDED)

        # Load A values for the current a_k column and relevant a_indices
        A_vals_tile = tl.load(a_ptr + a_indices * san + a_k * sak,
                             mask=a_indices < A_N,
                             other=0.0, cache_modifier='.ca')  # (BLOCK_N,)
        
        # Now compute the contribution using the identity formula
        # We want: result[m, a*B_N + b] += A[a, a_k] * sum_k(B[b, k] * X[m, a_k*B_K + k])
        # This is equivalent to: A[a, a_k] * (B @ X_slice)[b, m]
        
        # Compute B_padded @ X_slice_padded^T to get (B_N_PADDED, BLOCK_M)
        BX_padded = tl.dot(B_matrix_padded, tl.trans(X_slice_padded))  # (B_N_PADDED, BLOCK_M)
        
        # Now we need to extract BX_padded[b_indices, :] for each output position
        # Since we can't use variable indexing, we'll use a different approach:
        # Compute the contribution directly using broadcasting
        
        # Create masks for valid b_indices
        b_valid_mask = b_indices < B_N  # (BLOCK_N,)
        
        # For each valid b_index, we want BX_padded[b_index, :]
        # We can compute this using broadcasting and masking
        
        # Create a contribution matrix
        contribution = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Use broadcasting to compute the contribution
        # For each position (m, n), we want: A_vals_tile[n] * BX_padded[b_indices[n], m]
        # if b_indices[n] < B_N, else 0
        
        # Extract the relevant rows from BX_padded using gather-like operations
        # Since Triton doesn't support advanced indexing, we'll use a workaround
        
        # Method: Use the fact that b_indices are small integers
        # Compute contribution for each possible b value
        for b_val in range(B_N):
            # Create mask for positions where b_indices == b_val AND b_indices < B_N
            b_mask = (b_indices == b_val) & b_valid_mask  # (BLOCK_N,)
            
            # If any positions match this b_val, add their contribution
            if tl.sum(b_mask.to(tl.int32)) > 0:
                # Get BX_padded[b_val, :] - we know b_val < B_N so this is valid
                # Use a mask to extract this row safely
                b_row_mask = tl.arange(0, B_N_PADDED) == b_val  # (B_N_PADDED,)
                BX_row = tl.sum(tl.where(b_row_mask[:, None], BX_padded, 0.0), axis=0)  # (BLOCK_M,)
                
                # Add contribution: A_vals_tile[n] * BX_row[m] for positions where b_mask[n] is True
                contribution += tl.where(b_mask[None, :], 
                                       BX_row[:, None] * A_vals_tile[None, :], 
                                       0.0)
        
        acc += contribution

    # Add bias if present
    if bias_ptr is not None:
        bias_vec = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.)
        acc += bias_vec[None, :]

    # Store result
    tl.store(out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# ───────────────────────── Python wrapper ─────────────────────────

def kron_linear_forward_cpu(x, A, B, bias):
    """CPU implementation using true linear algebra identity: (A ⊗ B) * vec(X) = vec(B * X_reshaped * A^T)"""
    M, K = x.shape
    A_N, A_K = A.shape
    B_N, B_K = B.shape
    N = A_N * B_N
    
    # Verify dimensions
    assert K == A_K * B_K, f"Input dimension mismatch: K={K}, A_K*B_K={A_K*B_K}"
    
    # Reshape X from (M, A_K * B_K) to (M, A_K, B_K)
    X_reshaped = x.view(M, A_K, B_K)  # (M, A_K, B_K)
    
    # Apply the identity: (A ⊗ B) * vec(X) = vec(B * X_reshaped * A^T)
    # 
    # For each batch element m:
    # X_reshaped[m] is (A_K, B_K)
    # We want: B * X_reshaped[m] * A^T = (B_N, B_K) * (A_K, B_K) * (A_K, A_N)
    # This gives us (B_N, A_N) for each batch element
    
    # Method: Use batch matrix multiplication
    # X_reshaped: (M, A_K, B_K) -> transpose to (M, B_K, A_K) for bmm
    X_transposed = X_reshaped.transpose(1, 2)  # (M, B_K, A_K)
    
    # Step 1: B @ X_transposed = (B_N, B_K) @ (M, B_K, A_K) -> (M, B_N, A_K)
    # We need to broadcast B for batch multiplication
    B_expanded = B.unsqueeze(0).expand(M, -1, -1)  # (M, B_N, B_K)
    BX = torch.bmm(B_expanded, X_transposed)  # (M, B_N, A_K)
    
    # Step 2: BX @ A^T = (M, B_N, A_K) @ (A_K, A_N) -> (M, B_N, A_N)
    A_T_expanded = A.T.unsqueeze(0).expand(M, -1, -1)  # (M, A_K, A_N)
    result = torch.bmm(BX, A_T_expanded)  # (M, B_N, A_N)
    
    # Reshape to final output: (M, B_N, A_N) -> (M, A_N * B_N)
    # The Kronecker product layout is: (A ⊗ B)[i,j] where i = a*B_N + b, j = ak*B_K + bk
    # So output[m, a*B_N + b] = result[m, b, a]
    out = result.transpose(1, 2).contiguous().view(M, A_N * B_N)  # (M, N)
    
    if bias is not None:
        out += bias
    
    return out

def kron_linear_forward(x, A, B, bias):
    # Use CPU implementation if not on CUDA or if dimensions are problematic for Triton
    if not x.is_cuda or A.shape[1] * B.shape[1] != x.shape[1]:
        return kron_linear_forward_cpu(x, A, B, bias)
    
    M, K = x.shape
    A_N, A_K = A.shape
    B_N, B_K = B.shape
    N = A_N * B_N
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    # Handle bias properly - create a dummy bias tensor if None
    if bias is None:
        bias_ptr = None
        bias_stride = 0
    else:
        bias_ptr = bias
        bias_stride = bias.stride(0)

    kron_linear_kernel[grid](
        x, A, B, bias_ptr, out,
        M, K, N,
        A_N, A_K, B_N, B_K,
        x.stride(0), x.stride(1),
        A.stride(1), A.stride(0),
        B.stride(1), B.stride(0),
        bias_stride,
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
                    for _ in range(100):
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
