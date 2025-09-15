import argparse
import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, repeat
import itertools


def chunk_scan_triton(cb, x, dt, dA_cumsum, C, states, D):
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
    out, _ = _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D)
    return out


def ref_program(cb, x, dt, dA_cumsum, C, prev_states, D):
    """
    Argument:
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        C: (batch, seqlen, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    _, _, ngroups, _, _ = cb.shape
    batch, seqlen, nheads, headdim = x.shape
    # _, _, ngroups, dstate = B.shape
    # assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    # assert C.shape == B.shape
    # B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    cb = repeat(cb, "b c g l s -> b c (g h) l s", h=nheads // ngroups)
    # CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
    #                   rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = cb * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(
        C, "b (c l) h n -> b c l h n", c=nchunks), prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out


def get_configs_joint():
    """Joint调优：同时调优所有参数（简化版，避免卡住）"""
    configs = []
    
    # 论文中的关键配置 - 优先测试
    paper_configs = [
        # PT-Joint偏好：64×64 (更好的流水线)
        {'block_M': 64, 'block_N': 64, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 1},
        {'block_M': 64, 'block_N': 64, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 2},
        
        # PT-Decouple偏好：64×128 (更好的内存利用)  
        {'block_M': 64, 'block_N': 128, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 1},
        {'block_M': 64, 'block_N': 128, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 2},
    ]
    configs.extend(paper_configs)
    
    # 只使用最稳定的小配置，避免卡住
    stable_configs = [
        # 小配置组合
        {'block_M': 64, 'block_N': 32, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 1},
        {'block_M': 64, 'block_N': 32, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 2},
        {'block_M': 128, 'block_N': 32, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 1},
        {'block_M': 128, 'block_N': 32, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 2},
        {'block_M': 128, 'block_N': 64, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 1},
        {'block_M': 128, 'block_N': 64, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 2},
    ]
    
    # 添加稳定配置，避免重复
    for config in stable_configs:
        if config not in configs:
            configs.append(config)
    
    print(f"Joint配置数量: {len(configs)} (简化版，包含论文关键配置)")
    return configs


def get_configs_decouple_tiles():
    """Decouple调优阶段1：固定stage=2，调优tile参数（超级简化版）"""
    configs = [
        # 论文关键配置
        {'block_M': 64, 'block_N': 64, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 2},   # PT-Joint偏好
        {'block_M': 64, 'block_N': 128, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 2},  # PT-Decouple偏好
        # 基础配置
        # {'block_M': 64, 'block_N': 32, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 2},
        {'block_M': 128, 'block_N': 32, 'block_K': 64, 'block_Dstate': 128, 'num_stages': 2},
    ]
    
    return configs


def get_configs_decouple_stages(best_tile_config):
    """Decouple调优阶段2：固定tile参数，调优stage参数"""
    num_stages = [1, 2, 3]  # 只用低stage数避免卡住
    configs = [
        {
            'block_M': best_tile_config['block_M'], 
            'block_N': best_tile_config['block_N'], 
            'block_K': best_tile_config['block_K'], 
            'block_Dstate': best_tile_config['block_Dstate'], 
            'num_stages': s
        }
        for s in num_stages
    ]
    print(f"Stage配置数量: {len(configs)} (限制为低stage数)")
    return configs


# 保持原有函数用于向后兼容
def get_configs():
    """默认配置：简单的joint调优"""
    return get_configs_joint()


@autotune(configs=get_configs_joint(), warmup=10, rep=10)
@tilelang.jit(out_idx=[7])
def chunk_scan_fwd_joint(batch,
                         seqlen,
                         chunk_size,
                         ngroups,
                         nheads,
                         headdim,
                         dstate,
                         block_M=64,
                         block_N=64,
                         block_K=64,
                         block_Dstate=128,
                         num_stages=2,
                         threads=128):
    """Joint调优：同时调优所有参数"""
    return chunk_scan_fwd_kernel(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate,
                               block_M, block_N, block_K, block_Dstate, num_stages, threads)


@autotune(configs=get_configs_decouple_tiles(), warmup=10, rep=10)
@tilelang.jit(out_idx=[7])
def chunk_scan_fwd_decouple_tiles(batch,
                                  seqlen,
                                  chunk_size,
                                  ngroups,
                                  nheads,
                                  headdim,
                                  dstate,
                                  block_M=64,
                                  block_N=64,
                                  block_K=64,
                                  block_Dstate=128,
                                  num_stages=2,
                                  threads=128):
    """Decouple调优阶段1：固定stage=2，调优tile参数"""
    return chunk_scan_fwd_kernel(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate,
                               block_M, block_N, block_K, block_Dstate, num_stages, threads)


def chunk_scan_fwd_decouple(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate):
    """Decouple调优：分两阶段执行"""
    print("阶段1：固定stage=2，调优tile参数...")
    
    # 阶段1：调优tile参数
    tile_kernel = chunk_scan_fwd_decouple_tiles(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate)
    best_tile_config = tile_kernel.config
    
    print(f"阶段1最佳tile配置：{best_tile_config}")
    print("阶段2：固定tile参数，调优stage参数...")
    
    # 创建阶段2的配置
    stage_configs = get_configs_decouple_stages(best_tile_config)
    
    # 阶段2：调优stage参数
    @autotune(configs=stage_configs, warmup=10, rep=10)
    @tilelang.jit(out_idx=[7])
    def chunk_scan_fwd_decouple_stages(batch=batch,
                                       seqlen=seqlen,
                                       chunk_size=chunk_size,
                                       ngroups=ngroups,
                                       nheads=nheads,
                                       headdim=headdim,
                                       dstate=dstate,
                                       block_M=64,
                                       block_N=64,
                                       block_K=64,
                                       block_Dstate=128,
                                       num_stages=2,
                                       threads=128):
        return chunk_scan_fwd_kernel(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate,
                                   block_M, block_N, block_K, block_Dstate, num_stages, threads)
    
    stage_kernel = chunk_scan_fwd_decouple_stages(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate)
    print(f"阶段2最佳stage配置：{stage_kernel.config}")
    
    return stage_kernel


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[7])
def chunk_scan_fwd(batch,
                   seqlen,
                   chunk_size,
                   ngroups,
                   nheads,
                   headdim,
                   dstate,
                   block_M=64,
                   block_N=64,
                   block_K=64,
                   block_Dstate=128,
                   num_stages=2,
                   threads=128):
    return chunk_scan_fwd_kernel(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate,
                               block_M, block_N, block_K, block_Dstate, num_stages, threads)


def chunk_scan_fwd_kernel(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate,
                         block_M, block_N, block_K, block_Dstate, num_stages, threads):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504

    @T.prim_func
    def kernel_impl(cb: T.Tensor((batch, nchunks, ngroups, chunk_size, chunk_size), dtype), x: T.Tensor(
        (batch, seqlen, nheads, headdim), dtype), dt: T.Tensor(
            (batch, nheads, nchunks, chunk_size), dtype), dA_cumsum: T.Tensor(
                (batch, nheads, nchunks, chunk_size), dtype),
             C: T.Tensor((batch, seqlen, ngroups, dstate), dtype), prev_states: T.Tensor(
                 (batch, nchunks, nheads, headdim, dstate), dtype), D: T.Tensor(
                     (nheads), dtype), Output: T.Tensor((batch, seqlen, nheads, headdim), dtype)):
        with T.Kernel(
                nheads,
                T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N),
                batch * nchunks,
                threads=threads) as (bz, bx, by):
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
            cb_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.dyn")
            cb_local = T.alloc_fragment((block_M, block_K), dtype)
            dA_cs_k_shared = T.alloc_shared((block_K), dtype, scope="shared")
            dA_cs_k_local = T.alloc_fragment((block_K), accum_dtype)
            dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
            dt_shared = T.alloc_shared((block_K), dtype, scope="shared")
            dt_local = T.alloc_fragment((block_K), accum_dtype)
            x_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.dyn")
            dA_cs_m_shared = T.alloc_shared((block_M), dtype, scope="shared")
            scale_m_local = T.alloc_fragment((block_M), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)
            D_local = T.alloc_fragment((1), accum_dtype)
            x_residual_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.dyn")
            x_residual_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            batch_idx = by % batch
            chunk_idx = by // batch
            # m: chunk_size
            # n : headdim
            m_idx = bx // T.ceildiv(headdim, block_N)
            n_idx = bx % T.ceildiv(headdim, block_N)

            T.annotate_layout({
                acc_o_shared: tilelang.layout.make_swizzled_layout(acc_o_shared),
                cb_shared: tilelang.layout.make_swizzled_layout(cb_shared),
                x_residual_shared: tilelang.layout.make_swizzled_layout(x_residual_shared)
            })

            T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M:(m_idx + 1) * block_M],
                   dA_cs_m_shared)
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)

            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
            T.copy(
                C[batch_idx, chunk_idx * chunk_size + m_idx * block_M:chunk_idx * chunk_size +
                  (m_idx + 1) * block_M, bz // (nheads // ngroups), 0:block_Dstate], C_shared)
            T.copy(
                prev_states[batch_idx, chunk_idx, bz, n_idx * block_N:(n_idx + 1) * block_N,
                            0:block_Dstate], prev_state_shared)
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    cb[batch_idx, chunk_idx, bz // (nheads // ngroups),
                       m_idx * block_M:(m_idx + 1) * block_M, k * block_K:(k + 1) * block_K],
                    cb_shared)
                T.copy(cb_shared, cb_local)
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K:(k + 1) * block_K],
                       dA_cs_k_shared)
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i,
                             j] = cb_local[i,
                                           j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K:(k + 1) * block_K], dt_shared)
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(m_idx * block_M + i >= k * block_K + j,
                                                    cb_local[i, j], 0)
                T.copy(
                    x[batch_idx, chunk_idx * chunk_size + k * block_K:chunk_idx * chunk_size +
                      (k + 1) * block_K, bz, n_idx * block_N:(n_idx + 1) * block_N], x_shared)
                T.gemm(cb_local, x_shared, acc_o)

            D_local[0] = D[bz]
            T.copy(
                x[batch_idx, chunk_idx * chunk_size + m_idx * block_M:chunk_idx * chunk_size +
                  (m_idx + 1) * block_M, bz, n_idx * block_N:(n_idx + 1) * block_N],
                x_residual_shared)
            T.copy(x_residual_shared, x_residual_local)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] += x_residual_local[i, j] * D_local[0]

            T.copy(acc_o, acc_o_shared)
            T.copy(
                acc_o_shared,
                Output[batch_idx, chunk_idx * chunk_size + m_idx * block_M:chunk_idx * chunk_size +
                       (m_idx + 1) * block_M, bz, n_idx * block_N:(n_idx + 1) * block_N])

    return kernel_impl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--heads', type=int, default=64, help='heads')
    parser.add_argument('--groups', type=int, default=1, help='groups')
    parser.add_argument('--seq_len', type=int, default=2048, help='sequence length')
    parser.add_argument('--chunk_size', type=int, default=256, help='chunk size')
    parser.add_argument('--dim', type=int, default=64, help='dim')
    parser.add_argument('--dstate', type=int, default=128, help='dstate')
    parser.add_argument('--tune_mode', type=str, choices=['joint', 'decouple', 'none'], default='none', help='调优模式：joint(联合调优), decouple(分离调优), none(不调优)')
    parser.add_argument('--reproduce_paper', action='store_true', help='使用论文中的参数设置 (BS=64, SEQ=8k)')
    args = parser.parse_args()
    
    if args.reproduce_paper:
        # 论文参数：BS=64, SEQ=8k
        batch, heads, groups, seq_len, chunk_size, dim, dstate = 64, 64, 1, 8192, 256, 64, 128
        print("使用论文参数：BS=64, SEQ=8k (ChunkScan pattern)")
    else:
        # 用户指定参数
        batch, heads, groups, seq_len, chunk_size, dim, dstate = args.batch, args.heads, args.groups, args.seq_len, args.chunk_size, args.dim, args.dstate
        print(f"使用自定义参数：BS={batch}, SEQ={seq_len}")
    
    # 设置固定随机种子确保测试数据一致
    torch.manual_seed(42)
    total_flops = 2 * batch * seq_len * chunk_size * heads * dim * 0.5 + 2 * batch * seq_len * heads * dim * dstate

    if args.tune_mode == 'joint':
        print("\n=== Joint调优模式 ===")
        print("使用Joint调优模式...")
        kernel = chunk_scan_fwd_joint(batch, seq_len, chunk_size, groups, heads, dim, dstate)
        best_latency = kernel.latency
        best_config = kernel.config
        print(f"Joint最佳延迟: {best_latency:.4f} ms")
        print(f"Joint最佳TFlops: {total_flops / best_latency * 1e-9:.2f}")
        print(f"Joint最佳配置: {best_config}")
        
        # 论文复现分析
        if args.reproduce_paper:
            print("\n=== 论文复现分析 ===")
            print(f"论文PT-Joint结果: 6.981 ms (BS=64, SEQ=8k)")
            print(f"当前Joint结果: {best_latency:.4f} ms")
            print(f"性能比较: {6.981 / best_latency:.2f}x (>1表示当前更快)")
            tile_shape = f"{best_config['block_M']}×{best_config['block_N']}"
            print(f"最佳tile形状: {tile_shape}")
            if tile_shape == "64×64":
                print("✓ 与论文PT-Joint预期一致 (64×64, 更好的流水线)")
            elif tile_shape == "64×128":
                print("! 更接近PT-Decouple偏好 (64×128, 更好的内存利用)")
            else:
                print(f"? 发现了不同的最优配置: {tile_shape}")
        
    elif args.tune_mode == 'decouple':
        print("\n=== Decouple调优模式 ===")
        print("使用Decouple调优模式...")
        kernel = chunk_scan_fwd_decouple(batch, seq_len, chunk_size, groups, heads, dim, dstate)
        best_latency = kernel.latency
        best_config = kernel.config
        print(f"Decouple最佳延迟: {best_latency:.4f} ms")
        print(f"Decouple最佳TFlops: {total_flops / best_latency * 1e-9:.2f}")
        print(f"Decouple最佳配置: {best_config}")
        
        # 论文复现分析
        if args.reproduce_paper:
            print("\n=== 论文复现分析 ===")
            print(f"论文PT-Decouple结果: 12.150 ms (BS=64, SEQ=8k)")
            print(f"当前Decouple结果: {best_latency:.4f} ms")
            print(f"性能比较: {12.150 / best_latency:.2f}x (>1表示当前更快)")
            tile_shape = f"{best_config['block_M']}×{best_config['block_N']}"
            print(f"最佳tile形状: {tile_shape}")
            if tile_shape == "64×128":
                print("✓ 与论文PT-Decouple预期一致 (64×128, 更好的内存利用)")
            elif tile_shape == "64×64":
                print("! 更接近PT-Joint偏好 (64×64, 更好的流水线)")
            else:
                print(f"? 发现了不同的最优配置: {tile_shape}")
    
    else:
        print("\n=== 固定配置运行 ===")
        kernel = chunk_scan_fwd(
            batch,
            seq_len,
            chunk_size,
            groups,
            heads,
            dim,
            dstate,
            block_M=64,
            block_N=64,
            block_K=64,
            block_Dstate=128,
            num_stages=2,
            threads=128)
        profiler = kernel.get_profiler(tilelang.TensorSupplyType.Normal)
        profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
        print("All checks pass.")
        latency = profiler.do_bench(ref_program, warmup=500)
        print("Ref: {:.2f} ms".format(latency))
        print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = profiler.do_bench(warmup=500)
        print("固定配置延迟: {:.4f} ms".format(latency))
        print("固定配置TFlops: {:.2f}".format(total_flops / latency * 1e-9))
        
        if args.reproduce_paper:
            print("\n=== 固定配置vs论文对比 ===")
            print(f"论文Triton结果: 13.332 ms (BS=64, SEQ=8k)")
            print(f"当前固定配置: {latency:.4f} ms")
            print(f"性能比较: {13.332 / latency:.2f}x")


print("""
使用说明：

# 复现论文实验（推荐）
python example_mamba_chunk_scan.py --tune_mode joint --reproduce_paper
python example_mamba_chunk_scan.py --tune_mode decouple --reproduce_paper

# 默认参数测试
python example_mamba_chunk_scan.py --tune_mode joint
python example_mamba_chunk_scan.py --tune_mode decouple

# 快速测试
python example_mamba_chunk_scan.py --tune_mode none

关键观察点：
1. Joint vs Decouple的性能差异
2. 最佳配置的tile形状选择 (64×64 vs 64×128)
3. stage数对性能的影响
4. 与论文结果的趋势对比（不要求绝对数值匹配）

TileLang vs TVM-TL 框架差异可能导致结果不完全一致，但趋势应该相似。
""")
