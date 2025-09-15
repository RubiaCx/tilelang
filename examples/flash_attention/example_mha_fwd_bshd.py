import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial


def get_configs_joint():
    """Joint调优：同时调优所有参数"""
    configs = []
    
    # 关键配置 - 简单版本
    paper_configs = [
        {'block_M': 128, 'block_N': 128, 'num_stages': 1, 'threads': 256},
        {'block_M': 64, 'block_N': 64, 'num_stages': 1, 'threads': 128},
        {'block_M': 128, 'block_N': 64, 'num_stages': 1, 'threads': 256},
        {'block_M': 64, 'block_N': 128, 'num_stages': 1, 'threads': 128},
    ]
    configs.extend(paper_configs)
    
    # 其他稳定配置（简化版，适合单stage）
    stable_configs = [
        {'block_M': 64, 'block_N': 64, 'num_stages': 2, 'threads': 128},
        {'block_M': 128, 'block_N': 128, 'num_stages': 2, 'threads': 256},
    ]
    
    for config in stable_configs:
        if config not in configs:
            configs.append(config)
    
    print(f"Joint配置数量: {len(configs)}")
    return configs


def get_configs_decouple_tiles():
    """Decouple调优阶段1：固定stage=1，调优tile参数"""
    configs = [
        # 关键tile配置（简化版）
        {'block_M': 128, 'block_N': 128, 'num_stages': 1, 'threads': 256},  # 大tile
        {'block_M': 64, 'block_N': 64, 'num_stages': 1, 'threads': 128},    # 小tile
        {'block_M': 128, 'block_N': 64, 'num_stages': 1, 'threads': 256},   # 高M低N
        {'block_M': 64, 'block_N': 128, 'num_stages': 1, 'threads': 128},   # 低M高N
    ]
    print(f"Tile配置数量: {len(configs)} (固定stage=1)")
    return configs


def get_configs_decouple_stages(best_tile_config):
    """Decouple调优阶段2：固定tile参数，调优stage参数"""
    num_stages = [1, 2]  # 简化版只用1,2
    configs = [
        {
            'block_M': best_tile_config['block_M'],
            'block_N': best_tile_config['block_N'], 
            'num_stages': s,
            'threads': best_tile_config['threads']
        }
        for s in num_stages
    ]
    print(f"Stage配置数量: {len(configs)} (简化版)")
    return configs


# 保持原有函数用于向后兼容
def get_configs():
    return [{'block_M': 64, 'block_N': 64, 'num_stages': 1, 'threads': 128}]


@autotune(configs=get_configs_joint(), warmup=10, rep=10)
@tilelang.jit(out_idx=[3])
def flashattn_joint(batch,
                   heads,
                   seq_len,
                   dim,
                   is_causal,
                   block_M=64,
                   block_N=64,
                   num_stages=1,
                   threads=128):
    """Joint调优：同时调优所有参数"""
    return flashattn_kernel(batch, heads, seq_len, dim, is_causal, block_M, block_N, num_stages, threads)


@autotune(configs=get_configs_decouple_tiles(), warmup=10, rep=10)
@tilelang.jit(out_idx=[3])
def flashattn_decouple_tiles(batch,
                            heads,
                            seq_len,
                            dim,
                            is_causal,
                            block_M=64,
                            block_N=64,
                            num_stages=1,
                            threads=128):
    """Decouple调优阶段1：固定stage=1，调优tile参数"""
    return flashattn_kernel(batch, heads, seq_len, dim, is_causal, block_M, block_N, num_stages, threads)


def flashattn_decouple(batch, heads, seq_len, dim, is_causal):
    """Decouple调优：分两阶段执行"""
    print("阶段1：固定stage=1，调优tile参数...")
    
    # 阶段1：调优tile参数
    tile_kernel = flashattn_decouple_tiles(batch, heads, seq_len, dim, is_causal)
    best_tile_config = tile_kernel.config
    
    print(f"阶段1最佳tile配置：{best_tile_config}")
    print("阶段2：固定tile参数，调优stage参数...")
    
    # 创建阶段2的配置
    stage_configs = get_configs_decouple_stages(best_tile_config)
    
    # 阶段2：调优stage参数
    @autotune(configs=stage_configs, warmup=10, rep=10)
    @tilelang.jit(out_idx=[3])
    def flashattn_decouple_stages(batch=batch,
                                 heads=heads,
                                 seq_len=seq_len,
                                 dim=dim,
                                 is_causal=is_causal,
                                 block_M=64,
                                 block_N=64,
                                 num_stages=1,
                                 threads=128):
        return flashattn_kernel(batch, heads, seq_len, dim, is_causal, block_M, block_N, num_stages, threads)
    
    stage_kernel = flashattn_decouple_stages(batch, heads, seq_len, dim, is_causal)
    print(f"阶段2最佳stage配置：{stage_kernel.config}")
    
    return stage_kernel


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[3])
def flashattn(batch,
              heads,
              seq_len,
              dim,
              is_causal,
              block_M=64,
              block_N=64,
              num_stages=1,
              threads=128):
    return flashattn_kernel(batch, heads, seq_len, dim, is_causal, block_M, block_N, num_stages, threads)


def flashattn_kernel(batch, heads, seq_len, dim, is_causal, block_M, block_N, num_stages, threads):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.macro
    def MMA0(
        K: T.Tensor(shape, dtype),
        Q_shared: T.SharedBuffer([block_M, dim], dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                             -T.infinity(acc_s.dtype))
        else:
            T.clear(acc_s)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Tensor(shape, dtype),
        V_shared: T.SharedBuffer([block_M, dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
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
    def kernel_impl(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
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

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                    (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum,
                        logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

    return kernel_impl


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    return output


def main(
    batch: int = 8,
    heads: int = 32,
    seq_len: int = 4096,
    dim: int = 128,
    is_causal: bool = False,
    tune_mode: str = 'none',
    reproduce_paper: bool = False,
):
    if reproduce_paper:
        # FlashAttention论文参数：BS=64, SEQ=4k（简化版）
        batch, heads, seq_len, dim = 64, 32, 4096, 128
        print("使用论文参数：BS=64, H=32, SEQ=4k, D=128 (FlashAttention简化版)")
    else:
        print(f"使用自定义参数：BS={batch}, H={heads}, SEQ={seq_len}, D={dim}")
    
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    if tune_mode == 'joint':
        print("\n=== Joint调优模式 ===")
        print("使用Joint调优模式...")
        kernel = flashattn_joint(batch, heads, seq_len, dim, is_causal)
        best_latency = kernel.latency
        best_config = kernel.config
        print(f"Joint最佳延迟: {best_latency:.6f} ms")
        print(f"Joint最佳TFlops: {total_flops / best_latency * 1e-9:.2f}")
        print(f"Joint最佳配置: {best_config}")
        
        # 论文复现分析（简化版）
        if reproduce_paper:
            print("\n=== FlashAttention简化版调优分析 ===")
            tile_shape = f"{best_config['block_M']}×{best_config['block_N']}"
            print(f"最佳tile形状: {tile_shape}")
            print(f"最佳stage数: {best_config['num_stages']}")
            if tile_shape == "128×128":
                print("✓ 大tile配置 (更好的算术强度)")
            elif tile_shape == "64×64":
                print("! 小tile配置 (更好的并行度)")
            else:
                print(f"? 非对称tile配置: {tile_shape}")
        
    elif tune_mode == 'decouple':
        print("\n=== Decouple调优模式 ===")
        print("使用Decouple调优模式...")
        kernel = flashattn_decouple(batch, heads, seq_len, dim, is_causal)
        best_latency = kernel.latency
        best_config = kernel.config
        print(f"Decouple最佳延迟: {best_latency:.6f} ms")
        print(f"Decouple最佳TFlops: {total_flops / best_latency * 1e-9:.2f}")
        print(f"Decouple最佳配置: {best_config}")
        
        # 论文复现分析（简化版）
        if reproduce_paper:
            print("\n=== FlashAttention简化版调优分析 ===")
            tile_shape = f"{best_config['block_M']}×{best_config['block_N']}"
            print(f"最佳tile形状: {tile_shape}")
            print(f"最佳stage数: {best_config['num_stages']}")
            if best_config['block_M'] > best_config['block_N']:
                print("! 偏好高M低N配置 (内存合并友好)")
            elif best_config['block_M'] < best_config['block_N']:
                print("! 偏好低M高N配置 (并行度友好)")
            else:
                print("✓ 对称tile配置")
    
    else:
        print("\n=== 固定配置运行 ===")
        kernel = flashattn(
            batch,
            heads,
            seq_len,
            dim,
            is_causal,
            block_M=128,
            block_N=128,
            num_stages=1,
            threads=128)
        ref_program_processed = partial(ref_program, is_causal=is_causal)
        profiler = kernel.get_profiler()
        profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
        print("All checks pass.")
        latency = profiler.do_bench(ref_program_processed, warmup=500)
        print("Ref: {:.2f} ms".format(latency))
        print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = profiler.do_bench(warmup=500)
        print("固定配置延迟: {:.6f} ms".format(latency))
        print("固定配置TFlops: {:.6f}".format(total_flops / latency * 1e-9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FlashAttention简化版 Joint vs Decouple 调优')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--seq_len', type=int, default=4096, help='sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--is_causal', action='store_true', help='causal')
    parser.add_argument('--tune_mode', type=str, choices=['joint', 'decouple', 'none'], 
                       default='none', help='调优模式：joint(联合调优), decouple(分离调优), none(不调优)')
    parser.add_argument('--reproduce_paper', action='store_true', 
                       help='使用FlashAttention论文参数设置 (BS=64, H=32, SEQ=4k)')
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.tune_mode, args.reproduce_paper)


# print("""
# FlashAttention简化版调优使用说明：

# # 复现论文实验（推荐）
# python example_mha_fwd_bshd.py --tune_mode joint --reproduce_paper
# python example_mha_fwd_bshd.py --tune_mode decouple --reproduce_paper

# # 自定义参数测试
# python example_mha_fwd_bshd.py --tune_mode joint --batch 8 --seq_len 2048
# python example_mha_fwd_bshd.py --tune_mode decouple --batch 8 --seq_len 2048

# # 固定配置测试
# python example_mha_fwd_bshd.py --tune_mode none --reproduce_paper

# # Causal attention测试
# python example_mha_fwd_bshd.py --tune_mode joint --is_causal

# 关键观察点：
# 1. Joint vs Decouple的性能差异
# 2. 最佳tile形状选择 (128×128 vs 64×64 vs 非对称)
# 3. 简化版vs高级流水线版本的配置偏好差异
# 4. Causal vs Non-causal的配置偏好差异

# 简化版FlashAttention主要使用num_stages=1-2，适合理解基础tile优化。
# """)
