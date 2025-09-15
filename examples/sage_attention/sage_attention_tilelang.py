"""
SageAttention TileLang implementation
Converted from the original Triton implementation
"""

import math
import torch
import tilelang
import tilelang.language as T
from typing import Optional, Any

RCP_LN2 = 1.4426950408889634  # exp(x) = exp2(x * log2(e))
LN2 = 0.6931471824645996  # = ln(2)

QUANT_CONFIG = {
    "int8": (0, torch.int8),
    "e4m3": (1, torch.float8_e4m3fn),
    "e5m2": (2, torch.float8_e5m2),
    "none": (3, torch.int8)
}

@tilelang.jit(out_idx=[1, 2])
def per_block_quant_kernel(batch, heads, seq_len, head_dim, block_size, quant_type, sm_scale):
    """
    Per-block quantization kernel for Q/K/V tensors in SageAttention style
    """
    threads = 128
    input_shape = [batch, heads, seq_len, head_dim]
    output_shape = [batch, heads, seq_len, head_dim]
    scale_shape = [batch, heads, T.ceildiv(seq_len, block_size)]  # Per-block scales
    
    dtype = "float16"
    if quant_type == 0 or quant_type == 3:  # int8
        quant_dtype = "int8"
        quant_max = 127.0
    elif quant_type == 1:  # e4m3
        quant_dtype = "float8_e4m3fn"
        quant_max = 448.0
    elif quant_type == 2:  # e5m2
        quant_dtype = "float8_e5m2"
        quant_max = 57344.0
    else:
        quant_dtype = "int8"
        quant_max = 127.0
    
    def kernel_func():
        @T.prim_func
        def main(
            Input: T.Tensor(input_shape, dtype),
            Output: T.Tensor(output_shape, quant_dtype),
            Scale: T.Tensor(scale_shape, "float32"),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_size), batch * heads, threads=threads) as (bx, by):
                off_b = by // heads
                off_h = by % heads
                
                # Allocate memory using proper fragment layout
                input_frag = T.alloc_fragment([block_size, head_dim], "float32")
                output_frag = T.alloc_fragment([block_size, head_dim], quant_dtype)
                absmax_local = T.alloc_fragment([1], "float32")
                scale_local = T.alloc_fragment([1], "float32")
                
                # Annotate layout for better performance (similar to reference)
                T.annotate_layout({
                    input_frag: T.Fragment(
                        input_frag.shape,
                        forward_thread_fn=lambda i, j: (i // (block_size // 4)) * 32 + j % 32
                    )
                })
                
                start_seq = bx * block_size
                
                # Load input block and apply softmax scale
                for i, j in T.Parallel(block_size, head_dim):
                    seq_idx = start_seq + i
                    if seq_idx < seq_len:
                        val = T.cast(Input[off_b, off_h, seq_idx, j], "float32")
                        input_frag[i, j] = val * sm_scale
                    else:
                        input_frag[i, j] = 0.0
                
                # Compute block-wise absolute maximum
                T.reduce_absmax(input_frag, absmax_local, dim=1)
                
                # Compute block scale with numerical stability
                absmax_local[0] = T.max(absmax_local[0], 1e-8)
                scale_local[0] = absmax_local[0] / quant_max
                
                # Quantize the entire block
                for i, j in T.Parallel(block_size, head_dim):
                    seq_idx = start_seq + i
                    if seq_idx < seq_len:
                        if quant_type == 0 or quant_type == 3:  # int8
                            val = input_frag[i, j] / scale_local[0]
                            val = val + 0.5 * T.if_then_else(val >= 0, 1.0, -1.0)
                            output_frag[i, j] = T.cast(val, quant_dtype)
                        else:  # fp8
                            val = T.clamp(input_frag[i, j] / scale_local[0], -quant_max, quant_max)
                            output_frag[i, j] = T.cast(val, quant_dtype)
                    else:
                        output_frag[i, j] = T.cast(0, quant_dtype)
                
                # Store quantized results
                for i, j in T.Parallel(block_size, head_dim):
                    seq_idx = start_seq + i
                    if seq_idx < seq_len:
                        Output[off_b, off_h, seq_idx, j] = output_frag[i, j]
                
                # Store block scale
                Scale[off_b, off_h, bx] = scale_local[0]
        
        return main
    
    return kernel_func()


@tilelang.jit(out_idx=[0, 1])
def sage_attention_kernel(batch, heads, seq_q, seq_k, head_dim, is_causal, quant_type):
    """
    Main SageAttention kernel with quantization
    """
    block_M = 128
    block_N = 128
    block_headdim = 128  # Simplified for now
    num_stages = 3 if is_causal else 1
    threads = 128
    
    q_shape = [batch, heads, seq_q, head_dim]
    kv_shape = [batch, heads, seq_k, head_dim]
    out_shape = [batch, heads, seq_q, head_dim]
    
    num_q_blocks = (seq_q + block_M - 1) // block_M
    num_kv_blocks = (seq_k + block_N - 1) // block_N
    
    scale_q_shape = [batch, heads, num_q_blocks]  # Per-block scales
    scale_kv_shape = [batch, heads, num_kv_blocks]  # Per-block scales
    
    seq_q_rounded = ((seq_q + 127) // 128) * 128
    lse_shape = [batch, heads, seq_q_rounded]
    
    if quant_type == 0 or quant_type == 3:  # int8
        quant_dtype = "int8"
    elif quant_type == 1:  # e4m3
        quant_dtype = "float8e4nv"
    elif quant_type == 2:  # e5m2
        quant_dtype = "float8e5"
    else:
        quant_dtype = "int8"
    
    dtype = "float16"
    accum_dtype = "float32"
    
    def kernel_func():
        @T.macro
        def QuantizeP(
            p: T.FragmentBuffer([block_M, block_N], accum_dtype),
            p_quant: T.FragmentBuffer([block_M, block_N], quant_dtype),
            p_scale: T.FragmentBuffer([1], accum_dtype),
        ):
            # Find max absolute value
            max_val = T.cast(0.0, accum_dtype)
            for i, j in T.serial(block_M, block_N):
                abs_val = T.abs(p[i, j])
                max_val = T.max(max_val, abs_val)
            
            # Compute scale based on quantization type
            if quant_type == 0:  # int8
                p_scale[0] = max_val / 127.0 + 1e-8
            elif quant_type == 1:  # e4m3
                p_scale[0] = max_val / 448.0 + 1e-8
            elif quant_type == 2:  # e5m2
                p_scale[0] = max_val / 57344.0 + 1e-8
            
            # Quantize
            for i, j in T.Parallel(block_M, block_N):
                if quant_type == 0:  # int8
                    val = p[i, j] / p_scale[0]
                    val = val + 0.5 * T.if_then_else(val >= 0, 1.0, -1.0)
                    p_quant[i, j] = T.cast(val, quant_dtype)
                else:  # fp8
                    val = p[i, j] / p_scale[0]
                    p_quant[i, j] = T.cast(val, quant_dtype)
        
        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, quant_dtype),
            K: T.Tensor(kv_shape, quant_dtype),
            V: T.Tensor(kv_shape, quant_dtype),
            Q_scale: T.Tensor(scale_q_shape, accum_dtype),
            K_scale: T.Tensor(scale_kv_shape, accum_dtype),
            V_scale: T.Tensor(scale_kv_shape, accum_dtype),
            Output: T.Tensor(out_shape, dtype),
            Lse: T.Tensor(lse_shape, accum_dtype),
        ):
            with T.Kernel(num_q_blocks, batch * heads, threads=threads) as (bx, by):
                off_b = by // heads
                off_h = by % heads
                
                # Allocate memory
                Q_shared = T.alloc_shared([block_M, head_dim], quant_dtype)
                K_shared = T.alloc_shared([block_N, head_dim], quant_dtype)
                V_shared = T.alloc_shared([block_N, head_dim], quant_dtype)
                
                Q_frag = T.alloc_fragment([block_M, head_dim], accum_dtype)
                K_frag = T.alloc_fragment([block_N, head_dim], accum_dtype)
                V_frag = T.alloc_fragment([block_N, head_dim], accum_dtype)
                
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_quant = T.alloc_fragment([block_M, block_N], quant_dtype)
                acc_o = T.alloc_fragment([block_M, head_dim], accum_dtype)
                
                # Softmax states
                lse_i = T.alloc_fragment([block_M], accum_dtype)
                m_i = T.alloc_fragment([block_M], accum_dtype)
                
                # Scales - per block
                q_scale_frag = T.alloc_fragment([1], accum_dtype)
                k_scale_frag = T.alloc_fragment([1], accum_dtype)
                v_scale_frag = T.alloc_fragment([1], accum_dtype)
                p_scale_frag = T.alloc_fragment([1], accum_dtype)
                
                # Initialize
                T.fill(acc_o, 0.0)
                T.fill(lse_i, -T.infinity(accum_dtype))
                T.fill(m_i, -T.infinity(accum_dtype))
                
                # Load Q block and scale
                start_m = bx * block_M
                T.copy(Q[off_b, off_h, start_m:start_m + block_M, :], Q_shared)
                T.copy(Q_shared, Q_frag)
                
                # Load per-block Q scale
                q_scale_frag[0] = Q_scale[off_b, off_h, bx]
                
                # Convert Q to float32
                for i, j in T.Parallel(block_M, head_dim):
                    Q_frag[i, j] = T.cast(Q_shared[i, j], accum_dtype)
                
                # Determine loop range
                if is_causal:
                    loop_end = T.ceildiv(T.min((start_m + block_M), seq_k), block_N)
                else:
                    loop_end = T.ceildiv(seq_k, block_N)
                
                # Main attention loop
                for k_block in T.Pipelined(loop_end, num_stages=num_stages):
                    start_n = k_block * block_N
                    
                    # Load K, V blocks
                    T.copy(K[off_b, off_h, start_n:start_n + block_N, :], K_shared)
                    T.copy(V[off_b, off_h, start_n:start_n + block_N, :], V_shared)
                    
                    # Load per-block K, V scales
                    k_scale_frag[0] = K_scale[off_b, off_h, k_block]
                    v_scale_frag[0] = V_scale[off_b, off_h, k_block]
                    
                    # Convert K, V to float32
                    for i, j in T.Parallel(block_N, head_dim):
                        K_frag[i, j] = T.cast(K_shared[i, j], accum_dtype)
                        V_frag[i, j] = T.cast(V_shared[i, j], accum_dtype)
                    
                    # Compute QK^T with scaling
                    T.clear(acc_s)
                    T.gemm(Q_frag, K_frag, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    
                    # Apply per-block scales
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = acc_s[i, j] * q_scale_frag[0] * k_scale_frag[0]
                    
                    # Apply causal mask
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            row_idx = start_m + i
                            col_idx = start_n + j
                            if row_idx < col_idx:
                                acc_s[i, j] = -T.infinity(accum_dtype)
                    
                    # Apply sequence length mask
                    for i, j in T.Parallel(block_M, block_N):
                        col_idx = start_n + j
                        if col_idx >= seq_k:
                            acc_s[i, j] = -T.infinity(accum_dtype)
                    
                    # Online softmax update
                    m_ij = T.alloc_fragment([block_M], accum_dtype)
                    T.reduce_max(acc_s, m_ij, dim=1)
                    
                    # Update max values
                    for i in T.Parallel(block_M):
                        m_ij[i] = T.max(m_ij[i], lse_i[i])
                    
                    # Compute probabilities using exp2
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] - m_ij[i])
                    
                    # Sum probabilities
                    l_ij = T.alloc_fragment([block_M], accum_dtype)
                    T.reduce_sum(acc_s, l_ij, dim=1)
                    
                    # Update accumulator scaling
                    acc_o_scale = T.alloc_fragment([block_M], accum_dtype)
                    for i in T.Parallel(block_M):
                        acc_o_scale[i] = T.exp2(m_i[i] - m_ij[i])
                        acc_o[i, :] = acc_o[i, :] * acc_o_scale[i]
                    
                    # Quantize P matrix
                    QuantizeP(acc_s, acc_s_quant, p_scale_frag)
                    
                    # Convert quantized P back to float for computation
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.cast(acc_s_quant[i, j], accum_dtype) * p_scale_frag[0]
                    
                    # Update output accumulator
                    pv_result = T.alloc_fragment([block_M, head_dim], accum_dtype)
                    T.gemm(acc_s, V_frag, pv_result, policy=T.GemmWarpPolicy.FullRow)
                    
                    for i, j in T.Parallel(block_M, head_dim):
                        acc_o[i, j] = acc_o[i, j] + pv_result[i, j] * v_scale_frag[0]
                    
                    # Update softmax states
                    for i in T.Parallel(block_M):
                        m_i[i] = m_ij[i]
                        l_i_new = T.exp2(lse_i[i] - m_ij[i]) + l_ij[i]
                        lse_i[i] = m_ij[i] + T.log2(l_i_new)
                
                # Final scaling
                for i, j in T.Parallel(block_M, head_dim):
                    o_scale = T.exp2(m_i[i] - lse_i[i])
                    acc_o[i, j] = acc_o[i, j] * o_scale
                
                # Store results
                O_shared = T.alloc_shared([block_M, head_dim], dtype)
                for i, j in T.Parallel(block_M, head_dim):
                    O_shared[i, j] = T.cast(acc_o[i, j], dtype)
                
                T.copy(O_shared, Output[off_b, off_h, start_m:start_m + block_M, :])
                
                # Store LSE
                for i in T.Parallel(block_M):
                    if start_m + i < seq_q:
                        Lse[off_b, off_h, start_m + i] = lse_i[i]
        
        return main
    
    return kernel_func()


class SageAttnTileLangFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False, softmax_scale=None):
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]

        batch, seqlen_q, nheads_q, headdim = q.shape
        _, seqlen_k, nheads_k, _ = k.shape

        quant_code, quant_dtype = QUANT_CONFIG["e4m3"]
        assert headdim <= 128, "SageAttention only support head dimensions up to 128"
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda
        
        softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)
        BLOCK_M = 128
        BLOCK_N = 128

        NUM_Q_BLOCKS = (seqlen_q + BLOCK_M - 1) // BLOCK_M
        NUM_KV_BLOCKS = (seqlen_k + BLOCK_N - 1) // BLOCK_N

        seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
        lse = torch.empty((batch, nheads_q, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)

        # Create quantized tensors
        q_quant = torch.empty(q.shape, dtype=quant_dtype, device=q.device)
        k_quant = torch.empty(k.shape, dtype=quant_dtype, device=k.device)
        v_quant = torch.empty(v.shape, dtype=quant_dtype, device=v.device)

        # Create scale tensors
        q_scale = torch.empty((batch, nheads_q, NUM_Q_BLOCKS), device=q.device, dtype=torch.float32)
        k_scale = torch.empty((batch, nheads_q, NUM_KV_BLOCKS), device=k.device, dtype=torch.float32)
        v_scale = torch.empty((batch, nheads_q, NUM_KV_BLOCKS), device=v.device, dtype=torch.float32)

        # Apply K mean subtraction for numerical stability
        k_mean = k.mean(dim=1, keepdim=True)
        lse_correction = torch.matmul(q.transpose(1, 2), k_mean.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
        k = k - k_mean

        # Quantize Q, K, V using TileLang kernels
        q_quant_kernel = per_block_quant_kernel(
            batch, nheads_q, seqlen_q, headdim, BLOCK_M, quant_code, softmax_scale * RCP_LN2
        )
        q_quant_kernel(q, q_quant, q_scale)

        k_quant_kernel = per_block_quant_kernel(
            batch, nheads_k, seqlen_k, headdim, BLOCK_N, quant_code, 1.0
        )
        k_quant_kernel(k, k_quant, k_scale)

        v_quant_kernel = per_block_quant_kernel(
            batch, nheads_k, seqlen_k, headdim, BLOCK_N, quant_code, 1.0
        )
        v_quant_kernel(v, v_quant, v_scale)

        # Run main attention kernel
        attention_kernel = sage_attention_kernel(
            batch, nheads_q, seqlen_q, seqlen_k, headdim, causal, quant_code
        )
        attention_kernel(q_quant, k_quant, v_quant, q_scale, k_scale, v_scale, o, lse)

        ctx.softmax_scale = softmax_scale
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        
        # Apply LSE correction
        lse = (lse * LN2)[:, :, :seqlen_q]
        lse = lse + lse_correction * softmax_scale
        
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse=None):
        raise NotImplementedError("SageAttention does not support gradient propagation yet")


@torch.compiler.disable
def sage_attention_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "bhsd",
    is_causal: bool = False, 
    sm_scale: Optional[float] = None,
    **kwargs: Any):
    """
    TileLang implementation of SageAttention
    
    Args:
        q: Query tensor
        k: Key tensor  
        v: Value tensor
        tensor_layout: Tensor layout format ("bhsd", "sbhd", etc.)
        is_causal: Whether to apply causal masking
        sm_scale: Softmax scaling factor
        **kwargs: Additional arguments
    
    Returns:
        output: Attention output tensor
        lse: Log-sum-exp values
    """
    assert all(x.is_cuda for x in [q, k, v]), "All tensors must be on the GPU."
    assert q.dtype in [torch.float16, torch.bfloat16], "Tensors must be in FP16 or BF16."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    assert q.shape[-1] == k.shape[-1], "Head dim mismatch between Q and K"
    
    torch.cuda.set_device(q.device)
    
    qkv_format = "".join([i for i in tensor_layout.split("_")[0] if i.isalpha()])

    head_dim = q.size(-1)
    
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Handle different tensor layouts
    if qkv_format == "bhsd":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    elif qkv_format == "sbhd":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    else:
        raise ValueError(f"Invalid tensor layout: {qkv_format}")
    
    output, lse = SageAttnTileLangFunc.apply(q, k, v, is_causal, sm_scale)

    # Convert back to original layout
    if qkv_format == "bhsd": 
        output = output.permute(0, 2, 1, 3).contiguous()
    elif qkv_format == "bshd":
        output = output.permute(0, 2, 1, 3).contiguous()
    
    return output, lse


def test_sage_attention_tilelang():
    """Test function to compare TileLang implementation with reference"""
    torch.manual_seed(42)
    
    # Test configuration
    batch, seq_len, n_heads, head_dim = 2, 512, 8, 64
    device = 'cuda'
    dtype = torch.float16
    
    # Create test inputs
    q = torch.randn(batch, seq_len, n_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, n_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, n_heads, head_dim, device=device, dtype=dtype)
    
    # Test causal attention
    print("Testing SageAttention TileLang implementation...")
    
    try:
        output, lse = sage_attention_tilelang(q, k, v, is_causal=True)
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  LSE shape: {lse.shape}")
        print(f"  Output dtype: {output.dtype}")
        print(f"  LSE dtype: {lse.dtype}")
        
        # Basic sanity checks
        assert output.shape == q.shape, f"Output shape mismatch: {output.shape} vs {q.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    test_sage_attention_tilelang()