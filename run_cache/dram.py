import torch as th
import tilelang
import tilelang.language as T

N = 1 << 26# 约 256 MiB 的 float32 往返
VEC = 4      # 使用 float4 矢量化

@T.prim_func
def gmem_copy(src: T.Buffer((N,), 'float32'), dst: T.Buffer((N,), 'float32')):
    threads = 256
    with T.Kernel(T.ceildiv(N // VEC, threads), threads=threads) as bx:
        for i in T.Parallel(threads):
            base = (bx * threads + i) * VEC
            for v in T.vectorized(VEC):
                dst[base + v] = src[base + v]

jit = tilelang.compile(gmem_copy, target='auto')
src = th.empty(N, dtype=th.float32, device='cuda').uniform_()
dst = th.empty_like(src)

prof = jit.get_profiler()
ms = prof.do_bench(lambda: jit.torch_function(src, dst), warmup=25, rep=200, backend='event', return_mode='min')
bytes_moved = 2 * N * 4# 读+写
gbps = bytes_moved / (ms / 1000.0) / 1e9
print(f'DRAM effective bandwidth: {gbps:.1f} GB/s (min time {ms:.3f} ms)')
