"""
FP8 Training Kernels for Cinnamon

DeepSeek-style FP8 quantization with:
- Tile-wise scaling for activations (1x128)
- Block-wise scaling for weights (128x128)
- E4M3 format for both forward and backward
- Online quantization: scale = max(|x|) / 448
"""

import torch
import triton
import triton.language as tl

# FP8 E4M3 max representable value
FP8_E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def quantize_fp8_rowwise_kernel(
    x_ptr,           # Input tensor (BF16/FP32)
    x_fp8_ptr,       # Output FP8 tensor
    scale_ptr,       # Output scales (one per row-block)
    M, N,            # Matrix dimensions
    stride_xm, stride_xn,
    stride_fp8m, stride_fp8n,
    BLOCK_SIZE: tl.constexpr,  # 128 for DeepSeek-style
    ROUND_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr = 448.0,
):
    """
    Quantize activations with row-wise (1 x BLOCK_SIZE) scaling.
    Each row gets ceil(N / BLOCK_SIZE) scale factors.

    This is the "tile-wise" approach DeepSeek uses for activations.
    """
    # Program ID: (row, col_block)
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    # Column offsets for this block
    col_start = col_block * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load the block
    x_ptrs = x_ptr + row * stride_xm + cols * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Online scaling: max(|x|) / 448
    abs_max = tl.max(tl.abs(x))
    # Avoid division by zero and underflow - minimum scale prevents x/scale overflow
    abs_max = tl.maximum(abs_max, 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(abs_max / FP8_MAX)))
    else:
        scale = abs_max / FP8_MAX

    # Quantize with clamping to prevent overflow
    x_scaled = x / scale
    x_scaled = tl.minimum(tl.maximum(x_scaled, -FP8_MAX), FP8_MAX)
    x_fp8 = x_scaled.to(tl.float8e4nv)

    # Store quantized values
    fp8_ptrs = x_fp8_ptr + row * stride_fp8m + cols * stride_fp8n
    tl.store(fp8_ptrs, x_fp8, mask=mask)

    # Store scale (one per row-block)
    num_col_blocks = tl.cdiv(N, BLOCK_SIZE)
    scale_idx = row * num_col_blocks + col_block
    tl.store(scale_ptr + scale_idx, scale)


@triton.jit
def quantize_fp8_blockwise_kernel(
    x_ptr,           # Input tensor (BF16/FP32)
    x_fp8_ptr,       # Output FP8 tensor
    scale_ptr,       # Output scales (one per block)
    M, N,            # Matrix dimensions
    stride_xm, stride_xn,
    stride_fp8m, stride_fp8n,
    BLOCK_M: tl.constexpr,  # 128
    BLOCK_N: tl.constexpr,  # 128
    ROUND_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr = 448.0,
):
    """
    Quantize weights with block-wise (BLOCK_M x BLOCK_N) scaling.
    Each 128x128 block gets one scale factor.

    This is what DeepSeek uses for weights.
    """
    # Program ID: (row_block, col_block)
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    # Offsets
    row_start = row_block * BLOCK_M
    col_start = col_block * BLOCK_N
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Create 2D mask
    
    mask = (rows[:, None] < M) & (cols[None, :] < N)

    # Load 2D block
    x_ptrs = x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Online scaling: max(|x|) / 448 over entire block
    abs_max = tl.max(tl.abs(x))
    # Avoid division by zero and underflow - minimum scale prevents x/scale overflow
    abs_max = tl.maximum(abs_max, 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(abs_max / FP8_MAX)))
    else:
        scale = abs_max / FP8_MAX

    # Quantize with clamping to prevent overflow
    x_scaled = x / scale
    x_scaled = tl.minimum(tl.maximum(x_scaled, -FP8_MAX), FP8_MAX)
    x_fp8 = x_scaled.to(tl.float8e4nv)

    # Store quantized values
    fp8_ptrs = x_fp8_ptr + rows[:, None] * stride_fp8m + cols[None, :] * stride_fp8n
    tl.store(fp8_ptrs, x_fp8, mask=mask)

    # Store scale (one per 2D block)
    num_col_blocks = tl.cdiv(N, BLOCK_N)
    scale_idx = row_block * num_col_blocks + col_block
    tl.store(scale_ptr + scale_idx, scale)


@triton.jit
def dequantize_fp8_kernel(
    x_fp8_ptr,       # Input FP8 tensor
    scale_ptr,       # Input scales
    x_ptr,           # Output BF16 tensor
    M, N,
    stride_fp8m, stride_fp8n,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Dequantize FP8 back to BF16 using stored scales.
    """
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    row_start = row_block * BLOCK_M
    col_start = col_block * BLOCK_N
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    mask = (rows[:, None] < M) & (cols[None, :] < N)

    # Load FP8 values
    fp8_ptrs = x_fp8_ptr + rows[:, None] * stride_fp8m + cols[None, :] * stride_fp8n
    x_fp8 = tl.load(fp8_ptrs, mask=mask, other=0.0)

    # Load scale for this block
    num_col_blocks = tl.cdiv(N, BLOCK_N)
    scale_idx = row_block * num_col_blocks + col_block
    scale = tl.load(scale_ptr + scale_idx)

    # Dequantize
    x = x_fp8.to(tl.float32) * scale

    # Store
    x_ptrs = x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn
    tl.store(x_ptrs, x.to(tl.bfloat16), mask=mask)


# Python wrappers

def quantize_fp8_rowwise(
    x: torch.Tensor,
    block_size: int = 128,
    round_scale: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations with row-wise (1 x block_size) scaling.

    Args:
        x: Input tensor of shape (M, N) in BF16/FP32
        block_size: Number of elements per scale factor (default: 128)

    Returns:
        x_fp8: Quantized tensor in FP8 E4M3
        scales: Scale factors of shape (M, ceil(N / block_size))
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.ndim == 2, "Input must be 2D for now"

    M, N = x.shape
    num_col_blocks = (N + block_size - 1) // block_size

    # Allocate outputs
    x_fp8 = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((M, num_col_blocks), dtype=torch.float32, device=x.device)

    # Launch kernel
    grid = (M, num_col_blocks)
    quantize_fp8_rowwise_kernel[grid](
        x, x_fp8, scales,
        M, N,
        x.stride(0), x.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        BLOCK_SIZE=block_size,
        ROUND_SCALE=round_scale,
        FP8_MAX=448.0,  # Must pass explicitly for torch.compile compatibility
    )

    return x_fp8, scales


def quantize_fp8_blockwise(
    x: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    round_scale: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights with block-wise (block_m x block_n) scaling.

    Args:
        x: Input tensor of shape (M, N) in BF16/FP32
        block_m: Block size in M dimension (default: 128)
        block_n: Block size in N dimension (default: 128)

    Returns:
        x_fp8: Quantized tensor in FP8 E4M3
        scales: Scale factors of shape (ceil(M/block_m), ceil(N/block_n))
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.ndim == 2, "Input must be 2D"

    M, N = x.shape
    num_row_blocks = (M + block_m - 1) // block_m
    num_col_blocks = (N + block_n - 1) // block_n

    x_fp8 = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((num_row_blocks, num_col_blocks), dtype=torch.float32, device=x.device)

    grid = (num_row_blocks, num_col_blocks)
    quantize_fp8_blockwise_kernel[grid](
        x, x_fp8, scales,
        M, N,
        x.stride(0), x.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ROUND_SCALE=round_scale,
        FP8_MAX=448.0,  # Must pass explicitly for torch.compile compatibility
    )

    return x_fp8, scales


def dequantize_fp8_rowwise(x_fp8: torch.Tensor, scales: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantize FP8 tensor quantized with row-wise (1 x block_size) scaling.

    Args:
        x_fp8: FP8 tensor of shape (M, N)
        scales: Scale factors of shape (M, ceil(N / block_size))
        block_size: Block size used during quantization
    """
    M, N = x_fp8.shape
    num_blocks = scales.shape[1]
    pad = num_blocks * block_size - N
    if pad > 0:
        x_fp8 = torch.nn.functional.pad(x_fp8, (0, pad))
    x_fp8 = x_fp8.view(M, num_blocks, block_size).float()
    scales = scales.view(M, num_blocks, 1)
    x = (x_fp8 * scales).view(M, num_blocks * block_size)
    if pad > 0:
        x = x[:, :N]
    return x


def dequantize_fp8(x_fp8: torch.Tensor, scales: torch.Tensor, block_m: int = 128, block_n: int = 128) -> torch.Tensor:
    """
    Dequantize FP8 tensor back to BF16.

    Args:
        x_fp8: FP8 tensor of shape (M, N)
        scales: Scale factors
        block_m, block_n: Block sizes used during quantization

    Returns:
        x: Dequantized tensor in BF16
    """
    assert x_fp8.is_cuda, "Input must be on CUDA"

    M, N = x_fp8.shape
    num_row_blocks = (M + block_m - 1) // block_m
    num_col_blocks = (N + block_n - 1) // block_n

    x = torch.empty((M, N), dtype=torch.bfloat16, device=x_fp8.device)

    grid = (num_row_blocks, num_col_blocks)
    dequantize_fp8_kernel[grid](
        x_fp8, scales.flatten(), x,
        M, N,
        x_fp8.stride(0), x_fp8.stride(1),
        x.stride(0), x.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )

    return x


# =============================================================================
# Lightning Indexer Kernel (DeepSeek V3.2 style)
# =============================================================================

@triton.jit
def lightning_indexer_kernel(
    Q_ptr,          # Query tensor: (batch, seq_q, n_head, d_head)
    K_ptr,          # Key tensor: (batch, seq_k, d_head)
    W_ptr,          # Head weights: (batch, seq_q, n_head)
    Out_ptr,        # Output logits: (batch, seq_q, seq_k)
    batch_size, seq_q, seq_k, d_head,
    stride_qb, stride_qq, stride_qh, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_wb, stride_wq, stride_wh,
    stride_ob, stride_oq, stride_ok,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_HEAD: tl.constexpr,  # Compile-time head count for unrolling
):
    """
    Fused Lightning Indexer: computes I[b,q,k] = sum_h(ReLU(Q[b,q,h,:] · K[b,k,:]) * W[b,q,h])

    Optimizations:
    - N_HEAD as constexpr enables compile-time loop unrolling
    - K is loaded once per d_start, reused across all heads
    - FP32 accumulation, FP16 output for memory efficiency
    """
    # Program IDs
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Offsets
    q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    d_offs = tl.arange(0, BLOCK_D)

    q_mask = q_offs < seq_q
    k_mask = k_offs < seq_k

    # Final accumulator
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Per-head dot product accumulators
    # Use regular range and load weight per iteration to avoid constexpr indexing issues
    for h in range(N_HEAD):
        dot = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

        # Load weight for this head and query block
        w_ptrs = W_ptr + pid_b * stride_wb + q_offs[:, None] * stride_wq + h * stride_wh
        w_h = tl.load(w_ptrs, mask=q_mask[:, None], other=0.0)

        for d_start in range(0, d_head, BLOCK_D):
            d_idx = d_start + d_offs
            d_mask = d_idx < d_head

            # Load Q for this head
            q_ptrs = Q_ptr + pid_b * stride_qb + q_offs[:, None] * stride_qq + h * stride_qh + d_idx[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)

            # Load K (shared across heads - only depends on d_start)
            k_ptrs = K_ptr + pid_b * stride_kb + k_offs[:, None] * stride_kk + d_idx[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)

            dot += tl.dot(q.to(tl.float16), tl.trans(k.to(tl.float16)))

        # ReLU + weighted accumulation
        acc += tl.maximum(dot, 0.0) * w_h

    # Store
    out_ptrs = Out_ptr + pid_b * stride_ob + q_offs[:, None] * stride_oq + k_offs[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(tl.float16), mask=q_mask[:, None] & k_mask[None, :])


def lightning_indexer_fused(
    q: torch.Tensor,  # (batch, seq_q, n_head, d_head)
    k: torch.Tensor,  # (batch, seq_k, d_head)
    w: torch.Tensor,  # (batch, seq_q, n_head)
) -> torch.Tensor:
    """
    Fused Lightning Indexer kernel.

    Args:
        q: Query tensor of shape (batch, seq_q, n_head, d_head)
        k: Key tensor of shape (batch, seq_k, d_head)
    w: Head weights of shape (batch, seq_q, n_head)

    Returns:
        Logits tensor of shape (batch, seq_q, seq_k) in FP16
    """
    assert q.is_cuda and k.is_cuda, "Inputs must be on CUDA"
    batch, seq_q, n_head, d_head = q.shape
    _, seq_k, _ = k.shape

    out = torch.empty((batch, seq_q, seq_k), dtype=torch.float16, device=q.device)

    BLOCK_Q = 32
    BLOCK_K = 32
    BLOCK_D = min(64, d_head)

    grid = (batch, triton.cdiv(seq_q, BLOCK_Q), triton.cdiv(seq_k, BLOCK_K))

    lightning_indexer_kernel[grid](
        q, k, w, out,
        batch, seq_q, seq_k, d_head,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        N_HEAD=n_head,  # Passed as constexpr for compile-time unrolling
    )

    return out


# =============================================================================
# Grouped GEMM for MoE (processes all experts in one kernel)
# =============================================================================

@triton.jit
def grouped_gemm_kernel(
    # Input/output pointers
    A_ptr,           # Sorted input tokens: (total_tokens, d_in)
    B_ptr,           # Expert weights: (n_experts, d_out, d_in)
    C_ptr,           # Output: (total_tokens, d_out)
    # Expert boundaries
    expert_starts_ptr,  # Start index for each expert
    expert_ends_ptr,    # End index for each expert
    # Dimensions
    n_experts, d_in, d_out,
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Grouped GEMM: C[expert] = A[expert] @ B[expert].T

    Each expert processes its slice of sorted tokens.
    All experts run in parallel across different thread blocks.
    """
    # Program IDs
    pid_expert = tl.program_id(0)  # Which expert
    pid_m = tl.program_id(1)       # Which M tile within this expert
    pid_n = tl.program_id(2)       # Which N tile

    # Get this expert's token range
    expert_start = tl.load(expert_starts_ptr + pid_expert)
    expert_end = tl.load(expert_ends_ptr + pid_expert)
    expert_tokens = expert_end - expert_start

    # Early exit if this expert has no tokens or we're past its range
    if expert_tokens <= 0:
        return

    m_start = pid_m * BLOCK_M
    if m_start >= expert_tokens:
        return

    # Offsets within the expert's token range
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks
    mask_m = offs_m < expert_tokens
    mask_n = offs_n < d_out

    # Global row indices (offset by expert_start)
    global_m = expert_start + offs_m

    # Pointers
    a_ptrs = A_ptr + global_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + pid_expert * stride_be + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k_start in range(0, d_in, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < d_in

        # Load A tile: (BLOCK_M, BLOCK_K)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

        # Load B tile: (BLOCK_N, BLOCK_K) - transposed
        b = tl.load(b_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)

        # Compute: A @ B.T
        acc += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c_ptrs = C_ptr + global_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def moe_grouped_gemm(
    sorted_x: torch.Tensor,      # (total_tokens, d_in)
    weights: torch.Tensor,       # (n_experts, d_out, d_in)
    expert_starts: torch.Tensor, # (n_experts,)
    expert_ends: torch.Tensor,   # (n_experts,)
    max_expert_tokens: int = None,  # Max tokens assigned to any single expert
) -> torch.Tensor:
    """
    Grouped GEMM for MoE: each expert processes its slice of tokens.

    Args:
        sorted_x: Tokens sorted by expert assignment (total_tokens, d_in)
        weights: Expert weight matrices (n_experts, d_out, d_in)
        expert_starts: Start indices for each expert
        expert_ends: End indices for each expert
        max_expert_tokens: Max tokens for any expert (avoids over-launching kernels)

    Returns:
        Output tensor (total_tokens, d_out)
    """
    total_tokens, d_in = sorted_x.shape
    n_experts, d_out, _ = weights.shape

    # Output tensor
    out = torch.empty((total_tokens, d_out), dtype=torch.float16, device=sorted_x.device)

    # Block sizes
    BLOCK_M = 64  # Increased from 32 for better occupancy
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid: (n_experts, max_tiles_m, tiles_n)
    # Use actual max_expert_tokens to avoid launching ~n_experts× too many kernels
    if max_expert_tokens is None:
        # Fallback: assume even distribution with 2x buffer for imbalance
        max_expert_tokens = (total_tokens // n_experts) * 2 + BLOCK_M
    max_tiles_m = triton.cdiv(max_expert_tokens, BLOCK_M)
    grid = (
        n_experts,
        max_tiles_m,
        triton.cdiv(d_out, BLOCK_N),
    )

    grouped_gemm_kernel[grid](
        sorted_x, weights, out,
        expert_starts, expert_ends,
        n_experts, d_in, d_out,
        sorted_x.stride(0), sorted_x.stride(1),
        weights.stride(0), weights.stride(1), weights.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out


# =============================================================================
# FP8 Linear Layer (DeepSeek V3 style FP8 training)
# =============================================================================

@triton.jit
def fp8_gemm_kernel(
    # Inputs
    A_ptr,           # FP8 input: (M, K)
    A_scale_ptr,     # Input scales: (M, K // 128) - row-wise 1x128
    B_ptr,           # FP8 weights: (K, N)
    B_scale_ptr,     # Weight scales: (K // 128, N // 128) - block-wise 128x128
    C_ptr,           # Output: (M, N) in BF16
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Scale strides
    stride_as_m, stride_as_k,  # A scale strides (M, K//128)
    stride_bs_k, stride_bs_n,  # B scale strides (K//128, N//128)
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    FP8 GEMM: C = dequant(A) @ dequant(B)

    DeepSeek V3 style with:
    - A: row-wise 1x128 scaling (per row per 128 channels)
    - B: block-wise 128x128 scaling
    - FP32 accumulation
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # B scale block indices (128x128 blocks)
    scale_block_n = pid_n * BLOCK_N // 128

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K
        scale_block_k = k_start // 128

        # Load FP8 A tile: (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_fp8 = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

        # Load FP8 B tile: (BLOCK_K, BLOCK_N)
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_fp8 = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

        # Load A scales: row-wise, one per row per 128-column block
        # Shape: (BLOCK_M,) - each row in tile gets its own scale
        a_scale_ptrs = A_scale_ptr + offs_m * stride_as_m + scale_block_k * stride_as_k
        a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)  # (BLOCK_M,)

        # Load B scale: block-wise 128x128
        b_scale = tl.load(B_scale_ptr + scale_block_k * stride_bs_k + scale_block_n * stride_bs_n)

        # Dequantize with per-row scales for A, scalar for B
        a = a_fp8.to(tl.float32) * a_scales[:, None]  # broadcast row scales
        b = b_fp8.to(tl.float32) * b_scale

        # Matmul in FP32 (Triton accumulates in FP32 with float32 output_dtype)
        acc += tl.dot(a, b, out_dtype=tl.float32)

    # Store as BF16
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def fp8_matmul_dsv3(
    a: torch.Tensor,          # (M, K) FP8
    b: torch.Tensor,          # (K, N) FP8
    a_scales: torch.Tensor,   # (M, ceil(K/128)) - row-wise 1x128
    b_scales: torch.Tensor,   # (ceil(K/128), ceil(N/128)) - block-wise 128x128
) -> torch.Tensor:
    """
    FP8 matmul with DeepSeek V3 style scaling.

    - A (activations): row-wise 1x128 scaling (per token per 128 channels)
    - B (weights): block-wise 128x128 scaling

    Computes C = A @ B, accumulates in FP32, outputs BF16.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: {K} vs {K2}"

    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 128  # Use 128 for K to align with scale blocks

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fp8_gemm_kernel[grid](
        a, a_scales, b, b_scales, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        a_scales.stride(0), a_scales.stride(1),
        b_scales.stride(0), b_scales.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out


def _fp8_quantize_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 with per-tensor scaling for _scaled_mm."""
    # Per-tensor scale: max(|x|) / 448
    abs_max = x.abs().max().clamp(min=1e-12)
    scale = abs_max / 448.0
    x_scaled = (x / scale).clamp(-448, 448)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    # Ensure scale is on same device as input
    return x_fp8, scale.float().reshape(1).to(x.device)


def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad tensor dimensions to be divisible by multiple."""
    M, K = x.shape
    pad_m = (multiple - M % multiple) % multiple
    pad_k = (multiple - K % multiple) % multiple
    if pad_m > 0 or pad_k > 0:
        x = torch.nn.functional.pad(x, (0, pad_k, 0, pad_m))
    return x, (M, K)


def _unpad(x: torch.Tensor, orig_shape: tuple[int, int]) -> torch.Tensor:
    """Remove padding to restore original dimensions."""
    return x[:orig_shape[0], :orig_shape[1]]


class FP8MatmulFunction(torch.autograd.Function):
    """
    DeepSeek-V3 style FP8 matmul with fine-grained scaling.

    - Activations: row-wise 1x128 scaling
    - Weights: block-wise 128x128 scaling
    """

    @staticmethod
    def forward(ctx, x, weight, round_scale):
        x_shape = x.shape
        x_flat = x.reshape(-1, x_shape[-1]).contiguous()  # (M, K)
        M, K = x_flat.shape
        N = weight.shape[0]  # weight is (N, K)

        x_fp8, x_scale = quantize_fp8_rowwise(x_flat, block_size=128, round_scale=round_scale)
        w_fp8, w_scale = quantize_fp8_blockwise(weight.float(), block_m=128, block_n=128)
        w_fp8_t = w_fp8.t().contiguous()
        w_scale_t = w_scale.t().contiguous()

        if x.is_cuda:
            out = fp8_matmul_dsv3(x_fp8, w_fp8_t, x_scale, w_scale_t)
        else:
            x_deq = dequantize_fp8_rowwise(x_fp8, x_scale, block_size=128)
            w_deq = dequantize_fp8(w_fp8, w_scale, block_m=128, block_n=128)
            out = x_deq @ w_deq.t()

        ctx.save_for_backward(x_fp8, x_scale, w_fp8, w_scale)
        ctx.x_shape = x_shape
        ctx.round_scale = round_scale
        return out.reshape(*x_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        x_fp8, x_scale, w_fp8, w_scale = ctx.saved_tensors
        x_shape = ctx.x_shape
        round_scale = ctx.round_scale

        grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        grad_fp8, grad_scale = quantize_fp8_rowwise(grad_flat, block_size=128, round_scale=round_scale)

        if grad_output.is_cuda:
            grad_x = fp8_matmul_dsv3(grad_fp8, w_fp8, grad_scale, w_scale)
        else:
            grad_deq = dequantize_fp8_rowwise(grad_fp8, grad_scale, block_size=128)
            w_deq = dequantize_fp8(w_fp8, w_scale, block_m=128, block_n=128)
            grad_x = grad_deq @ w_deq

        if grad_output.is_cuda:
            grad_fp8_t, grad_scale_t = quantize_fp8_rowwise(
                grad_flat.t().contiguous(), block_size=128, round_scale=round_scale
            )
            x_deq = dequantize_fp8_rowwise(x_fp8, x_scale, block_size=128)
            x_fp8_block, x_scale_block = quantize_fp8_blockwise(
                x_deq, block_m=128, block_n=128, round_scale=False
            )
            grad_weight = fp8_matmul_dsv3(grad_fp8_t, x_fp8_block, grad_scale_t, x_scale_block)
        else:
            grad_deq = dequantize_fp8_rowwise(grad_fp8, grad_scale, block_size=128)
            x_deq = dequantize_fp8_rowwise(x_fp8, x_scale, block_size=128)
            grad_weight = grad_deq.t() @ x_deq

        return grad_x.reshape(x_shape), grad_weight, None


class FP8Linear(torch.nn.Module):
    """
    FP8 Linear layer with DeepSeek-V3 style fine-grained quantization.

    - Activations: row-wise 1x128 scaling
    - Weights: block-wise 128x128 scaling
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, round_scale: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.round_scale = round_scale

        # Master weights in FP32 for optimizer updates
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            out = FP8MatmulFunction.apply(x, self.weight, self.round_scale)
        else:
            out = torch.nn.functional.linear(x, self.weight, None)

        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, round_scale: bool = False) -> 'FP8Linear':
        """Convert an existing Linear layer to FP8Linear."""
        fp8_linear = cls(
            linear.in_features, linear.out_features, bias=linear.bias is not None, round_scale=round_scale
        )
        # Ensure weight is on same device/dtype as source
        fp8_linear.weight = torch.nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            fp8_linear.bias = torch.nn.Parameter(linear.bias.data.clone())
        return fp8_linear


def convert_to_fp8(model: torch.nn.Module, exclude_names: list = None) -> torch.nn.Module:
    """
    Convert Linear layers in a model to FP8Linear (DeepSeek V3 style).

    Per DeepSeek V3/V3.2, these components MUST stay in BF16/FP32:
    - embedding, lm_head (tied weights, need precision)
    - router (MoE gating, numerical stability)
    - norm layers (RMSNorm gamma)
    - indexer (DSA stability)

    Args:
        model: The model to convert
        exclude_names: Additional layer name patterns to exclude

    Returns:
        Model with FP8Linear layers (except excluded)
    """
    # Default exclusions per DeepSeek V3 paper Section 3.3.1
    # "we maintain the original precision (BF16/FP32) for: the embedding module,
    # the output head, MoE gating modules, normalization operators, and attention operators"
    default_exclude = [
        'embedding',    # Embedding module
        'lm_head',      # Output head
        'router',       # MoE gating modules
        'norm',         # Normalization operators (RMSNorm)
        'attn',         # ALL attention operators (w_dkv, w_uk, w_uv, w_dq, w_uq, w_qr, w_kr, w_out)
        'indexer',      # DSA indexer (implicit in attn, but explicit for safety)
    ]
    # FP8 is ONLY applied to MoE FFN layers: shared experts (w1, w2, w3) and routed experts

    exclude_names = (exclude_names or []) + default_exclude
    round_scale_names = [
        'w_out',  # Linear after attention
        'w1',     # SwiGLU up-projection
        'w2',     # SwiGLU up-projection
    ]

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check if this layer should be excluded
            if any(excl in name for excl in exclude_names):
                continue

            # Auto-exclude layers with dimensions < 128 (FP8 block size)
            if module.in_features < 128 or module.out_features < 128:
                print(f"[FP8] Auto-excluding {name}: shape ({module.out_features}, {module.in_features}) < 128")
                continue

            # Get parent module and attribute name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            # Replace with FP8Linear
            round_scale = any(key in name for key in round_scale_names)
            fp8_linear = FP8Linear.from_linear(module, round_scale=round_scale)
            setattr(parent, attr_name, fp8_linear)

    return model


# =============================================================================
# TRUE Sparse Attention Kernels
# O(L*K) compute and memory instead of O(L^2)
# =============================================================================

@triton.jit
def sparse_attention_fwd_kernel(
    # Inputs
    Q_ptr, K_ptr, V_ptr, Idx_ptr,
    # Outputs
    O_ptr, LSE_ptr, AttnW_ptr,
    # Dimensions
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr,
    K_sparse: tl.constexpr, D_qk: tl.constexpr, D_v: tl.constexpr,
    # Strides for Q: [B, H, L, D_qk]
    stride_qb, stride_qh, stride_ql, stride_qd,
    # Strides for K: [B, H, L, D_qk]
    stride_kb, stride_kh, stride_kl, stride_kd,
    # Strides for V: [B, H, L, D_v]
    stride_vb, stride_vh, stride_vl, stride_vd,
    # Strides for Idx: [B, L, K_sparse]
    stride_ib, stride_il, stride_ik,
    # Strides for O: [B, H, L, D_v]
    stride_ob, stride_oh, stride_ol, stride_od,
    # Strides for LSE: [B, H, L]
    stride_lseb, stride_lseh, stride_lsel,
    # Strides for AttnW: [B, H, L, K_sparse]
    stride_awb, stride_awh, stride_awl, stride_awk,
    # Scale factor
    scale,
    # Block sizes
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    TRUE sparse attention forward kernel with FlashAttention-style online softmax.

    Processes BLOCK_Q query positions per kernel launch to reduce launch overhead.
    Grid: (B, cdiv(L, BLOCK_Q), H) instead of (B, L, H)

    Complexity: O(L * K_sparse * D) instead of O(L^2 * D)
    """
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Base pointers for this batch/head
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    d_offs = tl.arange(0, BLOCK_D)
    q_mask = d_offs < D_qk
    v_mask = d_offs < D_v

    # Process BLOCK_Q queries in this kernel
    for q_idx in range(BLOCK_Q):
        t = pid_q * BLOCK_Q + q_idx
        # Guard all work inside t < L check (Triton doesn't support continue)
        if t < L:
            # Load query for this position
            q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql
            idx_base = Idx_ptr + pid_b * stride_ib + t * stride_il

            q = tl.load(q_base + d_offs * stride_qd, mask=q_mask, other=0.0).to(tl.float32)

            # PASS 1: Online softmax - compute max and sum
            global_max = float('-inf')
            global_sum = 0.0

            for k_start in range(0, K_sparse, BLOCK_K):
                k_offs = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_offs < K_sparse

                idx = tl.load(idx_base + k_offs * stride_ik, mask=k_mask, other=0)
                valid_mask = k_mask & (idx <= t)

                k_gathered = tl.load(
                    k_base + idx[:, None] * stride_kl + d_offs[None, :] * stride_kd,
                    mask=valid_mask[:, None] & q_mask[None, :],
                    other=0.0
                ).to(tl.float32)

                scores = tl.sum(q[None, :] * k_gathered, axis=1) * scale
                scores = tl.where(valid_mask, scores, float('-inf'))

                block_max = tl.max(scores)
                block_max = tl.where(block_max == float('-inf'), global_max, block_max)
                new_max = tl.maximum(global_max, block_max)

                correction = tl.exp(global_max - new_max)
                global_sum = global_sum * correction

                exp_scores = tl.exp(scores - new_max)
                exp_scores = tl.where(valid_mask, exp_scores, 0.0)
                global_sum = global_sum + tl.sum(exp_scores)

                global_max = new_max

            has_valid = global_sum > 0
            safe_sum = tl.where(has_valid, global_sum, 1.0)
            safe_max = tl.where(has_valid, global_max, 0.0)
            lse = safe_max + tl.log(safe_sum)

            # PASS 2: Compute output with normalized weights
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)

            for k_start in range(0, K_sparse, BLOCK_K):
                k_offs = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_offs < K_sparse

                idx = tl.load(idx_base + k_offs * stride_ik, mask=k_mask, other=0)
                valid_mask = k_mask & (idx <= t)

                k_gathered = tl.load(
                    k_base + idx[:, None] * stride_kl + d_offs[None, :] * stride_kd,
                    mask=valid_mask[:, None] & q_mask[None, :],
                    other=0.0
                ).to(tl.float32)

                scores = tl.sum(q[None, :] * k_gathered, axis=1) * scale
                scores = tl.where(valid_mask, scores, float('-inf'))

                attn_weights = tl.exp(scores - lse)
                attn_weights = tl.where(valid_mask, attn_weights, 0.0)

                # Store attention weights
                aw_base = AttnW_ptr + pid_b * stride_awb + pid_h * stride_awh + t * stride_awl
                tl.store(aw_base + k_offs * stride_awk, attn_weights.to(tl.float32), mask=k_mask)

                v_gathered = tl.load(
                    v_base + idx[:, None] * stride_vl + d_offs[None, :] * stride_vd,
                    mask=valid_mask[:, None] & v_mask[None, :],
                    other=0.0
                ).to(tl.float32)

                acc = acc + tl.sum(attn_weights[:, None] * v_gathered, axis=0)

            # Store output and LSE
            o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh + t * stride_ol
            tl.store(o_base + d_offs * stride_od, acc.to(tl.float32), mask=v_mask)

            lse_ptr = LSE_ptr + pid_b * stride_lseb + pid_h * stride_lseh + t * stride_lsel
            tl.store(lse_ptr, lse)


@triton.jit
def sparse_attention_bwd_kernel(
    # Gradients
    dO_ptr, O_ptr, LSE_ptr,
    # Inputs
    Q_ptr, K_ptr, V_ptr, Idx_ptr,
    # Output gradients
    dQ_ptr, dK_ptr, dV_ptr,
    # Dimensions
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr,
    K_sparse: tl.constexpr, D_qk: tl.constexpr, D_v: tl.constexpr,
    # Strides for dO/O: [B, H, L, D_v]
    stride_dob, stride_doh, stride_dol, stride_dod,
    # Strides for LSE: [B, H, L]
    stride_lseb, stride_lseh, stride_lsel,
    # Strides for Q/dQ: [B, H, L, D_qk]
    stride_qb, stride_qh, stride_ql, stride_qd,
    # Strides for K/dK: [B, H, L, D_qk]
    stride_kb, stride_kh, stride_kl, stride_kd,
    # Strides for V/dV: [B, H, L, D_v]
    stride_vb, stride_vh, stride_vl, stride_vd,
    # Strides for Idx: [B, L, K_sparse]
    stride_ib, stride_il, stride_ik,
    # Scale factor
    scale,
    # Block sizes
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Backward kernel for sparse attention with BLOCK_Q queries per launch.
    Grid: (B, cdiv(L, BLOCK_Q), H) - 64x fewer launches than per-query.
    """
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Base pointers for this batch/head
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    dk_base = dK_ptr + pid_b * stride_kb + pid_h * stride_kh
    dv_base = dV_ptr + pid_b * stride_vb + pid_h * stride_vh

    d_offs = tl.arange(0, BLOCK_D)
    d_mask_v = d_offs < D_v
    d_mask_qk = d_offs < D_qk

    # Process BLOCK_Q queries
    for q_idx in range(BLOCK_Q):
        t = pid_q * BLOCK_Q + q_idx
        if t < L:
            # Per-query bases
            do_base = dO_ptr + pid_b * stride_dob + pid_h * stride_doh + t * stride_dol
            o_base = O_ptr + pid_b * stride_dob + pid_h * stride_doh + t * stride_dol
            q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql
            idx_base = Idx_ptr + pid_b * stride_ib + t * stride_il
            dq_base = dQ_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql

            # Load LSE
            lse_ptr = LSE_ptr + pid_b * stride_lseb + pid_h * stride_lseh + t * stride_lsel
            lse = tl.load(lse_ptr)

            # Load dO, O, compute delta
            dO = tl.load(do_base + d_offs * stride_dod, mask=d_mask_v, other=0.0).to(tl.float32)
            O = tl.load(o_base + d_offs * stride_dod, mask=d_mask_v, other=0.0).to(tl.float32)
            delta = tl.sum(dO * O)

            # Load query
            q = tl.load(q_base + d_offs * stride_qd, mask=d_mask_qk, other=0.0).to(tl.float32)

            # dQ accumulator
            dq = tl.zeros([BLOCK_D], dtype=tl.float32)

            for k_start in range(0, K_sparse, BLOCK_K):
                k_offs = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_offs < K_sparse

                idx = tl.load(idx_base + k_offs * stride_ik, mask=k_mask, other=0)
                valid_mask = k_mask & (idx <= t)

                # Gather keys and compute scores
                k_gathered = tl.load(
                    k_base + idx[:, None] * stride_kl + d_offs[None, :] * stride_kd,
                    mask=valid_mask[:, None] & d_mask_qk[None, :], other=0.0
                ).to(tl.float32)

                scores = tl.sum(q[None, :] * k_gathered, axis=1) * scale
                scores = tl.where(valid_mask, scores, float('-inf'))

                p = tl.exp(scores - lse)
                p = tl.where(valid_mask, p, 0.0)

                # Gather values
                v_tile = tl.load(
                    v_base + idx[:, None] * stride_vl + d_offs[None, :] * stride_vd,
                    mask=valid_mask[:, None] & d_mask_v[None, :], other=0.0
                ).to(tl.float32)

                # dp = p * (dO · V - delta)
                dO_v = tl.sum(dO[None, :] * v_tile, axis=1)
                dp = p * (dO_v - delta)
                dp = tl.where(valid_mask, dp, 0.0)

                # dQ += sum_k(dp[k] * K[idx[k]]) * scale
                dq += tl.sum(dp[:, None] * k_gathered, axis=0) * scale

                # dK and dV via atomic adds (one element at a time)
                # Load q/dO elements individually - Triton can't index tensors with scalars
                for d_idx in tl.static_range(BLOCK_D):
                    if d_idx < D_qk:
                        q_elem = tl.load(q_base + d_idx * stride_qd)
                        dk_col = dp * q_elem * scale  # [BLOCK_K] * scalar
                        tl.atomic_add(dk_base + idx * stride_kl + d_idx * stride_kd, dk_col, mask=valid_mask)
                    if d_idx < D_v:
                        dO_elem = tl.load(do_base + d_idx * stride_dod)
                        dv_col = p * dO_elem  # [BLOCK_K] * scalar
                        tl.atomic_add(dv_base + idx * stride_vl + d_idx * stride_vd, dv_col, mask=valid_mask)

            # Store dQ
            tl.store(dq_base + d_offs * stride_qd, dq.to(tl.float32), mask=d_mask_qk)


class SparseAttentionFunction(torch.autograd.Function):
    """
    Autograd function for TRUE sparse attention.

    O(L * K) compute and memory instead of O(L^2).
    """

    @staticmethod
    def forward(ctx, q, k, v, idx, scale):
        """
        Forward pass for sparse attention.

        Args:
            q: [B, H, L, D_qk] query tensor
            k: [B, H, L, D_qk] key tensor
            v: [B, H, L, D_v] value tensor
            idx: [B, L, K] sparse attention indices
            scale: attention scale factor

        Returns:
            output: [B, H, L, D_v] attention output
            attn_weights: [B, H, L, K] attention weights (unnormalized exp)
        """
        B, H, L, D_qk = q.shape
        D_v = v.shape[-1]
        K_sparse = idx.shape[-1]
        input_dtype = q.dtype

        # Allocate outputs (float32 for kernel, convert at end)
        output = torch.empty(B, H, L, D_v, device=q.device, dtype=torch.float32)
        lse = torch.empty(B, H, L, device=q.device, dtype=torch.float32)
        attn_weights = torch.empty(B, H, L, K_sparse, device=q.device, dtype=torch.float32)

        # Block sizes - BLOCK_Q reduces kernel launches by 64x
        BLOCK_Q = 64
        BLOCK_K = min(64, triton.next_power_of_2(K_sparse))  # Process more K per iteration
        BLOCK_D = triton.next_power_of_2(max(D_qk, D_v))  # No cap - PoPE needs 192

        # Ensure tensors are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        idx = idx.contiguous()

        # Launch kernel - grid reduced by BLOCK_Q
        grid = (B, triton.cdiv(L, BLOCK_Q), H)
        sparse_attention_fwd_kernel[grid](
            q, k, v, idx,
            output, lse, attn_weights,
            B, H, L, K_sparse, D_qk, D_v,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            idx.stride(0), idx.stride(1), idx.stride(2),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
            scale,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, idx, output, lse)
        ctx.scale = scale
        ctx.BLOCK_K = BLOCK_K
        ctx.BLOCK_D = BLOCK_D
        ctx.input_dtype = input_dtype

        return output.to(input_dtype), attn_weights

    @staticmethod
    def backward(ctx, grad_output, grad_attn_weights):
        q, k, v, idx, output, lse = ctx.saved_tensors
        scale = ctx.scale
        input_dtype = ctx.input_dtype

        B, H, L, D_qk = q.shape
        D_v = v.shape[-1]
        K_sparse = idx.shape[-1]

        # Allocate gradients (float32 for kernel, convert at end)
        grad_q = torch.zeros(B, H, L, D_qk, device=q.device, dtype=torch.float32)
        grad_k = torch.zeros(B, H, L, D_qk, device=q.device, dtype=torch.float32)
        grad_v = torch.zeros(B, H, L, D_v, device=q.device, dtype=torch.float32)

        # Ensure contiguous
        grad_output = grad_output.contiguous()

        # Launch backward kernel with BLOCK_Q (64x fewer launches)
        BLOCK_Q = 64
        grid = (B, triton.cdiv(L, BLOCK_Q), H)
        sparse_attention_bwd_kernel[grid](
            grad_output, output, lse,
            q, k, v, idx,
            grad_q, grad_k, grad_v,
            B, H, L, K_sparse, D_qk, D_v,
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            idx.stride(0), idx.stride(1), idx.stride(2),
            scale,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=ctx.BLOCK_K,
            BLOCK_D=ctx.BLOCK_D,
            num_warps=4,
        )

        return grad_q.to(input_dtype), grad_k.to(input_dtype), grad_v.to(input_dtype), None, None


def sparse_attention(q, k, v, idx, scale=None):
    """
    TRUE sparse attention - O(L*K) instead of O(L^2).

    Only computes attention scores for K selected positions per query,
    rather than computing full L×L scores and masking.

    Args:
        q: [B, H, L, D_qk] queries (bfloat16)
        k: [B, H, L, D_qk] keys (bfloat16)
        v: [B, H, L, D_v] values (bfloat16)
        idx: [B, L, K] indices of selected positions per query (int64)
        scale: optional scale factor (default: 1/sqrt(D_qk))

    Returns:
        output: [B, H, L, D_v] attention output
        attn_weights: [B, H, L, K] attention weights (for auxiliary loss)
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return SparseAttentionFunction.apply(q, k, v, idx, scale)
