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
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(x, kron_result)

        
        # # Add debug points
        # tl.debug_barrier()  # Force synchronization
        # tl.device_print("BLOCK_N: {}", BLOCK_N)
        # tl.device_print("BLOCK_K: {}", BLOCK_K)
        # tl.device_print("N: {}, K: {}", N, K)
        # # tl.device_print("First offs_n: {}", offs_n[0])
        # # tl.device_print("First offs_k: {}", offs_k[0])
        # #tl.device_print("First mask value: {}", (offs_n[None, :] < N)[0, 0])
        
        # tl.device_print("Loaded weights shape: {}", int(w.shape[0]))
        # tl.device_print("Mask values: {}", (offs_n[None, :] < N) & (offs_k[:, None] < K))
        
        # # Debug specific values
        # tl.device_print("w_ptr: {}", w_ptr)
        # #tl.device_print("swn: {}, swk: {}", swn, swk)

        # # Debug memory access
        # #tl.device_print("Memory access pattern: {}", w_ptr + offs_n[None, :] * swn + offs_k[:, None] * swk)

        # # Debug mask conditions
        # tl.device_print("offs_n < N: {}", (offs_n[None, :] < N))
        # tl.device_print("offs_k < K: {}", (offs_k[:, None] < K))
    






    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]
    tl.store(out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))