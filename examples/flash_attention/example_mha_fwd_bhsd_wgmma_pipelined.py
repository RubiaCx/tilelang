import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial


def get_configs():
    iter_params = dict(block_M=[64, 128], block_N=[32, 64, 128], threads=[128, 256])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[3], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def flashattn(batch,
              heads,
              seq_q,
              seq_kv,
              dim,
              is_causal,
              block_M=128,
              block_N=128,
              num_stages=2,
              threads=256):
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
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # in the first ceil_div(kBlockM, kBlockN) steps.
        # for i in T.Parallel(block_M):
        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
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

            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_kv, block_N), T.ceildiv(
                        (bx + 1) * block_M +
                        past_len, block_N)) if is_causal else T.ceildiv(seq_kv, block_N))

            # for k in T.Pipelined(
            #         loop_range,
            #         num_stages=num_stages,
            #         order=[-1, 0, 3, 1, -1, 2],
            #         stage=[-1, 0, 0, 1, -1, 1],
            #         group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]):
            #     MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
            #     Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
            #     Rescale(acc_o, scores_scale)
            #     MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
            for k in T.Pipelined(
                    loop_range,
                    # # BASELINE
                    # #     Max error: 6.104e-04
                    # #     Mean absolute error: 1.468e-05
                    # #     L2 norm of error: 2.363e-01
                    # #     Cosine similarity: 1.000
                    # #     Best latency: 5.072 ms
                    # #     Best TFlops: 433.553 TFlops
                    # #     Best config: {'block_M': 128, 'block_N': 128, 'threads': 256}
                    # num_stages=2,
                    # order=[-1, 0, 3, 1, -1, 2],
                    # stage=[-1, 0, 0, 1, -1, 1],
                    # group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]],

                    # # 合并 Softmax 与 Rescale 为一大组，减少组间屏障数量
                    # #     Max error: 8.545e-04
                    # #     Mean absolute error: 1.468e-05
                    # #     L2 norm of error: 2.364e-01
                    # #     Cosine similarity: 1.000
                    # #     Best latency: 5.024 ms
                    # #     Best TFlops: 437.684 TFlops
                    # #     Best config: {'block_M': 128, 'block_N': 128, 'threads': 256}
                    # num_stages=2,
                    # order=[-1, 0, 2, -1, 1],
                    # stage=[-1, 0, 0, -1, 1],
                    # group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [12], [13]],

                    # # 细粒度拆分，Softmax→Rescale→SV 在同一 stage=1，order 递增，Producer 组保持 -1
                    # #     Max error: 5.649e-01
                    # #     Mean absolute error: 2.303e-02
                    # #     L2 norm of error: 3.368e+02
                    # #     Cosine similarity: 0.364
                    # #     Best latency: 5.426 ms
                    # num_stages=2,
                    # order=[-1, 0, 1, 2, 3, -1, 4],
                    # stage=[-1, 0, 0, 1, 1, -1, 1],
                    # group=[[0], [1], [2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]],
                 
                    # # 探索 triple-buffer
                    # #     Max error: 4.978e-01
                    # #     Mean absolute error: 2.285e-02
                    # #     L2 norm of error: 3.342e+02
                    # #     Cosine similarity: 0.366
                    # #     Best latency: 5.356 ms
                    # #     Best TFlops: 410.550 TFlops
                    # #     Best config: {'block_M': 128, 'block_N': 128, 'threads': 256}
                    num_stages=3,
                    order=[-1, 0, 1, 2, -1, 3],
                    stage=[-1, 0, 1, 1, -1, 2],
                    group=[[0], [1, 2], [3,4,5,6,7,8,9,10], [11], [12], [13]],
                    ):
                # MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared) # 0
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i + past_len
                        k_idx = k * block_N + j
                        acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype)) # 1
                else:
                    T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow) # 2
                # Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                T.copy(scores_max, scores_max_prev) # 3
                T.fill(scores_max, -T.infinity(accum_dtype)) # 4
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)  # 5
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale) # 6
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale) # 7
                T.reduce_sum(acc_s, scores_sum, dim=1) # 8
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i] # 9
                T.copy(acc_s, acc_s_cast) # 10
                # Rescale(acc_o, scores_scale)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i] # 11
                # MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared) # 12
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow) # 13
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

    return main


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_q = Q.size(2)
        seq_kv = K.size(2)
        mask = torch.tril(torch.ones(seq_q, seq_kv, device=scores.device), seq_kv - seq_q)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bhkd->bhqd', attention_weights, V)
    return output


def main(
    batch: int = 1,
    heads: int = 32,
    seq_q: int = 256,
    seq_kv: int = 256,
    dim: int = 128,
    is_causal: bool = False,
    tune: bool = False,
):
    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    if (not tune):
        kernel = flashattn(
            batch,
            heads,
            seq_q,
            seq_kv,
            dim,
            is_causal,
            block_M=128,
            block_N=128,
            num_stages=2,
            threads=256)
        ref_program_processed = partial(ref_program, is_causal=is_causal)

        profiler = kernel.get_profiler()
        # Print numerical differences between TileLang kernel and reference
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            Q_t = torch.randn(batch, heads, seq_q, dim, device=device, dtype=torch.float16)
            K_t = torch.randn(batch, heads, seq_kv, dim, device=device, dtype=torch.float16)
            V_t = torch.randn(batch, heads, seq_kv, dim, device=device, dtype=torch.float16)
            ref_out = ref_program_processed(Q_t, K_t, V_t)
            # Try to invoke kernel to get output
            try:
                tl_out = kernel(Q_t, K_t, V_t)
            except Exception:
                # Fallback: some kernels require preallocated output as last arg
                tl_out = torch.empty_like(ref_out)
                kernel(Q_t, K_t, V_t, tl_out)
            err = (tl_out.float() - ref_out.float())
            max_err = err.abs().max().item()
            mean_abs_err = err.abs().mean().item()
            l2_err = torch.norm(err).item()
            a = tl_out.float().reshape(-1)
            b = ref_out.float().reshape(-1)
            denom = (torch.norm(a) * torch.norm(b)).clamp_min(1e-12)
            cos_sim = (torch.dot(a, b) / denom).item()
            print(f"Max error: {max_err:.3e}")
            print(f"Mean absolute error: {mean_abs_err:.3e}")
            print(f"L2 norm of error: {l2_err:.3e}")
            print(f"Cosine similarity: {cos_sim:.3f}")
        ref_latency = profiler.do_bench(ref_program_processed, warmup=100)
        print("Ref: {:.2f} ms".format(ref_latency))
        # print("Ref: {:.2f} TFlops".format(total_flops / ref_latency * 1e-9))
        latency = profiler.do_bench(warmup=100)
        print("Tile-lang: {:.2f} ms".format(latency))
        # print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        print("speedup: {:.2f}".format(ref_latency / latency))
    else:
        kernel = flashattn(batch, heads, seq_q, seq_kv, dim, is_causal)
        best_latency = kernel.latency
        best_config = kernel.config
        ref_latency = kernel.ref_latency
        print("Best latency: {:.3f} ms".format(best_latency))
        print("Best TFlops: {:.3f} TFlops".format(total_flops / best_latency * 1e-9))
        print(f"Best config: {best_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--seq_q', type=int, default=4096, help='query sequence length')
    parser.add_argument('--seq_kv', type=int, default=4096, help='key/value sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--is_causal', action='store_true', help='causal')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_q, args.seq_kv, args.dim, args.is_causal, args.tune)
