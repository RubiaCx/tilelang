import tilelang.language as T
from tilelang.tools import Analyzer
from tilelang.carver.arch import CUDA
from tilelang.carver.arch import CDNA
import torch

M = N = K = 2048


def kernel(
    block_M=None,
    block_N=None,
    block_K=None,
    num_stages=None,
    thread_num=None,
    enable_rasteration=None,
):
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def matmul(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    transpose_B=True,
                )
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return matmul


def main():
    my_func = kernel(128, 128, 32, 3, 128, True)

    cuda_device = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
    result = Analyzer.analysis(my_func, cuda_device)
    print(result)

    # Basic correctness check: FLOPs (converted to TFLOPs for readability)
    analyzed_flops = result.total_flops
    expected_flops = 2 * M * N * K
    analyzed_tflops_total = analyzed_flops / 1e12
    expected_tflops_total = expected_flops / 1e12
    print(f"Analyzed total work: {analyzed_tflops_total:.3f} TFlops")
    print(f"Expected total work: {expected_tflops_total:.3f} TFlops")

    # Convert estimated_time to Python float if it's a TVM / TileLang FloatImm
    try:
        est_time = float(result.estimated_time)
    except TypeError:
        est_time = float(getattr(result.estimated_time, "value", result.estimated_time))

    # Derived performance metrics
    if est_time > 0:
        achieved_tflops = analyzed_flops / est_time / 1e12
        # total_global_bytes 可能也是 TVM 的 Imm，先转成 Python float
        try:
            total_bytes = float(result.total_global_bytes)
        except TypeError:
            total_bytes = float(
                getattr(result.total_global_bytes, "value", result.total_global_bytes)
            )
        achieved_bandwidth = total_bytes / est_time / 1e9
    else:
        achieved_tflops = 0.0
        achieved_bandwidth = 0.0

    print(f"Achieved TFLOPS: {achieved_tflops:.3f} TFlops")
    print(f"Achieved Bandwidth: {achieved_bandwidth:.3f} GB/s")


if __name__ == "__main__":
    main()
