import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial
from h2o import H2O


def get_configs():
    iter_params = dict(block_M=[128], block_N=[128], num_stages=[2], threads=[256])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def h2oattn(batch,
              heads,
              seq_q,
              seq_kv,
              dim,
              is_causal,
              block_M=64,
              block_N=64,
              num_stages=1,
              threads=128):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = "float16"
    accum_dtype = "float"

    past_len = seq_kv - seq_q
    assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

    @T.macro
    def MMA0(
        K: T.Tensor(kv_shape, dtype),
        Q_shared: T.SharedBuffer([block_M, dim], dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                q_idx = bx * block_M + i + past_len
                k_idx = k * block_N + j
                acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
        else:
            # We shall fill -inf for OOB positions
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Tensor(kv_shape, dtype),
        V_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            scores_max: T.FragmentBuffer([block_M], accum_dtype),
            scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
            scores_sum: T.FragmentBuffer([block_M], accum_dtype),
            logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)

        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
            H2OScore: T.Tensor([batch, heads, seq_kv], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            h2o_col = T.alloc_fragment([block_N], accum_dtype)

            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_kv, block_N), T.ceildiv(
                        (bx + 1) * block_M +
                        past_len, block_N)) if is_causal else T.ceildiv(seq_kv, block_N))

            # 第一遍：标准 FlashAttention 前向，计算输出
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum,
                        logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

            # 第二遍：在保留的 scores_max / logsum 基础上，计算每个 key 的概率列和
            # 这里直接对每个 (i, j) 的概率做原子加到 H2OScore 上，避免额外的列缓冲区归约。
            for k in range(loop_range):
                # 重新计算当前 block 的打分 acc_s（带 mask 的 QK^T）
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
 
                # 对本 block 所有 query 的概率，直接累加到对应 key 的 H2OScore 上
                for i, j in T.Parallel(block_M, block_N):
                    key_idx = k * block_N + j
                    # 被 mask 掉的位置在 acc_s 中是 -inf，对应概率为 0
                    prob_ij = T.if_then_else(
                        acc_s[i, j] == -T.infinity(acc_s.dtype), 0,
                        T.exp2(acc_s[i, j] * scale - scores_max[i] * scale) / logsum[i])
                    T.atomic_add(H2OScore[bz, by, key_idx], prob_ij)

    return main


def ref_program(Q, K, V, is_causal):
    """
    参考实现：直接调用 h2o.py 里的 H2O 模型。

    输入 Q/K/V 形状为 [batch, heads, seq, dim]（与 TileLang kernel 一致），
    H2O 期望的形状为 [batch, seq, heads, dim]，因此这里做一次转置。
    """
    batch, heads, seq_q, dim = Q.shape
    device = Q.device
    dtype = Q.dtype

    # H2O 中内部始终使用 causal mask，这里忽略 is_causal 参数，
    # 只保证数值逻辑与 h2o.py 完全一致。
    model = H2O(kv_head_num=heads, head_num=heads, head_dim=dim).to(device=device)
    model = model.to(dtype=dtype)
    model.eval()

    # [B, H, L, D] -> [B, L, H, D]
    Q_h2o = Q.transpose(1, 2).contiguous()
    K_h2o = K.transpose(1, 2).contiguous()
    V_h2o = V.transpose(1, 2).contiguous()

    with torch.no_grad():
        out, h2o_score = model(Q_h2o, K_h2o, V_h2o)
        # H2O 输出为 [B, L, H, D]，转回 [B, H, L, D] 与 TileLang 对齐
        out = out.transpose(1, 2).contiguous()
        # h2o_score: [B, kv_heads, kv_len]，这里 kv_heads == heads
        h2o_score = h2o_score
    return out, h2o_score


def main(
    batch: int = 1,
    heads: int = 1,
    seq_q: int = 256,
    seq_kv: int = 256,
    dim: int = 64,
    tune: bool = False,
):
    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total_flops = 2 * flops_per_matmul * 0.5
    if (not tune):
        kernel = h2oattn(
            batch,
            heads,
            seq_q,
            seq_kv,
            dim,
            True,
            block_M=64,
            block_N=64,
            num_stages=1,
            threads=128)
        ref_program_processed = partial(ref_program, is_causal=True)

        # 手动构造一组输入，显式比较 Output 和 h2o_score 的误差
        Q = torch.randn(batch, seq_q, heads, dim, dtype=torch.float16, device="cuda")
        K = torch.randn(batch, seq_kv, heads, dim, dtype=torch.float16, device="cuda")
        V = torch.randn(batch, seq_kv, heads, dim, dtype=torch.float16, device="cuda")

        with torch.no_grad():
            out_ref, h2o_score_ref = ref_program_processed(Q, K, V)

            out_tl = torch.empty_like(out_ref)
            h2o_score_tl = torch.zeros(batch, seq_kv, heads, dtype=torch.float32, device="cuda")
            kernel(Q, K, V, out_tl, h2o_score_tl)

            diff_out = (out_tl - out_ref).abs()
            max_err_out = diff_out.max().item()

            h2o_score_ref_bt = h2o_score_ref.to(h2o_score_tl.dtype)
            diff_h2o = (h2o_score_tl - h2o_score_ref_bt).abs()
            max_err_h2o = diff_h2o.max().item()

        print(f"Output max abs error: {max_err_out:.3e}")
        print(
            "Output allclose:",
            torch.allclose(out_tl, out_ref, rtol=0.01, atol=0.01),
        )
        print(f"h2o_score max abs error: {max_err_h2o:.3e}")
        print(
            "h2o_score allclose:",
            torch.allclose(h2o_score_tl, h2o_score_ref_bt, rtol=0.01, atol=0.01),
        )
    else:
        kernel = h2oattn(batch, heads, seq_q, seq_kv, dim)
        best_latency = kernel.latency
        best_config = kernel.config
        ref_latency = kernel.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--seq_q', type=int, default=4096, help='query sequence length')
    parser.add_argument('--seq_kv', type=int, default=4096, help='key/value sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_q, args.seq_kv, args.dim, args.tune)
