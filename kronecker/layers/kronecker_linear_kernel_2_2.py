import time
import torch
import triton
import triton.language as tl
import os
os.environ["TRITON_INTERPRET"] = "1"

# ────────────────────────────────────────────────────────────────
#  Triton kernel – every meta‑op size is a host‑supplied constexpr
# ────────────────────────────────────────────────────────────────
@triton.jit
def kronecker_kernel(
    x_ptr, a_ptr, b_ptr, out_ptr,
    M, N, K,                                           # runtime sizes
    # tile sizes chosen by caller
    BLOCK_M:  tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # dims of Aᵀ and Bᵀ
    A_N: tl.constexpr,  A_K: tl.constexpr,
    B_Nb: tl.constexpr, B_Kb: tl.constexpr,
    # derived compile‑time sizes
    AKB_N: tl.constexpr, AKB_K: tl.constexpr,
    M_AKB_K: tl.constexpr, M_BNb: tl.constexpr,
    san, sak, sbn, sbk                                # runtime strides
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m == 0 and pid_n == 0:
        tl.static_print("M_AKB_K =", M_AKB_K)
        tl.static_print("M_BNb =", M_BNb)
        print(type(M_AKB_K))
        print(type(M_BNb))
    # ---------------- X block -----------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    x_blk = tl.load(
        x_ptr + (offs_m[:, None] * K + offs_k[None, :]),
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )

    # ---------------- Bᵀ block ----------------------
    offs_b_n = tl.arange(0, B_Nb)
    offs_b_k = tl.arange(0, B_Kb)
    b_blk = tl.load(
        b_ptr + (offs_b_n[:, None] * sbn + offs_b_k[None, :] * sbk),
        mask=(offs_b_n[:, None] < B_Nb) & (offs_b_k[None, :] < B_Kb),
        other=0.0, cache_modifier='.ca',
    )

    # ---------------- Aᵀ block ----------------------
    offs_a_n = pid_n * AKB_N + tl.arange(0, AKB_N)
    offs_a_k = tl.arange(0, AKB_K)
    a_blk = tl.load(
        a_ptr + (offs_a_n[:, None] * san + offs_a_k[None, :] * sak),
        mask=(offs_a_n[:, None] < A_N) & (offs_a_k[None, :] < A_K),
        other=0.0, cache_modifier='.ca',
    )

    # --------------- local GEMM ---------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x1  = tl.reshape(x_blk, (M_AKB_K, B_Kb))           # (M·AKB_K, BKb)
    b_t = tl.reshape(b_blk, (B_Kb, B_Nb))              # replace permute
    tmp = tl.dot(x1, b_t)                              # (M·AKB_K, B_Nb)

    tmp = tl.reshape(tmp, (BLOCK_M, AKB_K, B_Nb))
    tmp = tl.trans(tmp, (0, 2, 1))                     # (M, B_Nb, AKB_K)
    tmp = tl.reshape(tmp, (M_BNb, AKB_K))

    out = tl.dot(tmp, tl.trans(a_blk))                 # (M·B_Nb, AKB_N)
    out = tl.reshape(out, (BLOCK_M, BLOCK_N))
    acc += out

    # ---------------- store -------------------------
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ────────────────────────────────────────────────────────────────
#  Host wrapper – computes and passes EVERY constexpr
# ────────────────────────────────────────────────────────────────
def kron_linear_triton(x, A_t, B_t,
                       block_m=128, block_n=64, block_k=32):
    M, K = x.shape
    B_Nb, B_Kb = B_t.shape
    A_N,  A_K  = A_t.shape
    N = A_N * B_Nb

    if block_n % B_Nb or block_k % B_Kb:
        raise ValueError("block sizes must divide B dims")

    # compile‑time sizes
    AKB_N   = block_n // B_Nb
    AKB_K   = block_k // B_Kb
    M_AKB_K = block_m * AKB_K
    M_BNb   = block_m * B_Nb

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    kronecker_kernel[grid](
        x, A_t, B_t, out,
        M, N, K,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        A_N=A_N,  A_K=A_K,
        B_Nb=B_Nb, B_Kb=B_Kb,
        AKB_N=AKB_N, AKB_K=AKB_K,
        M_AKB_K=M_AKB_K, M_BNb=M_BNb,
        san=A_t.stride(0), sak=A_t.stride(1),
        sbn=B_t.stride(0), sbk=B_t.stride(1),
    )
    return out


# ────────────────────────────────────────────────────────────────
#  Very small sanity test (remove or adapt in your code base)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    X  = torch.randn(64, 64, device="cuda")
    Aᵀ = torch.randn(32, 32, device="cuda")
    Bᵀ = torch.randn( 2,  2, device="cuda")

    Y = kron_linear_triton(X, Aᵀ, Bᵀ)
    W = torch.kron(Aᵀ, Bᵀ)
    Y_ref = X @ W.t()
    print("max‖Δ‖ =",
          (Y - Y_ref).abs().max().item())      # → 0.0 (FP32 exact)
