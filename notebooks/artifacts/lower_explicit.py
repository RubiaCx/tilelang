# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel(K_desc: T.handle("uint8x128", "grid_constant"), Output_desc: T.handle("uint8x128", "grid_constant"), Q_desc: T.handle("uint8x128", "grid_constant"), V_desc: T.handle("uint8x128", "grid_constant")):
        T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": 40960, "target": T.target({"arch": "sm_90", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 4, "blockIdx.y": 1, "blockIdx.z": 1, "threadIdx.x": 256, "threadIdx.y": 1, "threadIdx.z": 1}, "tir.is_global_func": T.bool(True), "tir.kernel_launch_params": ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tir.use_dyn_shared_memory"], "tir.noalias": True})
        acc_s_cast = T.handle("float16", "local")
        acc_s_cast_1 = T.decl_buffer((32,), "float16", data=acc_s_cast, scope="local")
        scores_sum = T.handle("float32", "local")
        scores_sum_1 = T.decl_buffer((2,), data=scores_sum, scope="local")
        scores_scale = T.handle("float32", "local")
        scores_scale_1 = T.decl_buffer((2,), data=scores_scale, scope="local")
        scores_max_prev = T.handle("float32", "local")
        scores_max_prev_1 = T.decl_buffer((2,), data=scores_max_prev, scope="local")
        acc_s = T.handle("float32", "local")
        acc_s_1 = T.decl_buffer((32,), data=acc_s, scope="local")
        scores_max = T.handle("float32", "local")
        scores_max_1 = T.decl_buffer((2,), data=scores_max, scope="local")
        logsum = T.handle("float32", "local")
        logsum_1 = T.decl_buffer((2,), data=logsum, scope="local")
        acc_o = T.handle("float32", "local")
        acc_o_1 = T.decl_buffer((32,), data=acc_o, scope="local")
        bx = T.launch_thread("blockIdx.x", 4)
        buf_dyn_shmem = T.allocate([40960], "uint8", "shared.dyn")
        acc_o = T.allocate([32], "float32", "local")
        logsum = T.allocate([2], "float32", "local")
        scores_max = T.allocate([2], "float32", "local")
        acc_s = T.allocate([32], "float32", "local")
        scores_max_prev = T.allocate([2], "float32", "local")
        scores_scale = T.allocate([2], "float32", "local")
        scores_sum = T.allocate([2], "float32", "local")
        acc_s_cast = T.allocate([32], "float16", "local")
        by = T.launch_thread("blockIdx.y", 1)
        bz = T.launch_thread("blockIdx.z", 1)
        tx = T.launch_thread("threadIdx.x", 256)
        T.create_barriers(9)
        if T.tl_shuffle_elect(0):
            T.call_extern("handle", "tl::prefetch_tma_descriptor", Q_desc)
            T.call_extern("handle", "tl::prefetch_tma_descriptor", K_desc)
            T.call_extern("handle", "tl::prefetch_tma_descriptor", V_desc)
            T.call_extern("handle", "tl::prefetch_tma_descriptor", Output_desc)
            T.ptx_init_barrier_thread_count(T.get_mbarrier(0), 1)
            T.ptx_init_barrier_thread_count(T.get_mbarrier(1), 1)
            T.ptx_init_barrier_thread_count(T.get_mbarrier(2), 1)
            T.ptx_init_barrier_thread_count(T.get_mbarrier(3), 1)
            T.ptx_init_barrier_thread_count(T.get_mbarrier(4), 1)
            T.ptx_init_barrier_thread_count(T.get_mbarrier(5), 1)
            T.ptx_init_barrier_thread_count(T.get_mbarrier(6), 1)
            T.ptx_init_barrier_thread_count(T.get_mbarrier(7), 1)
            T.ptx_init_barrier_thread_count(T.get_mbarrier(8), 1)
        T.tvm_storage_sync("shared")
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        T.attr([128, 128], "kWarpSpecializationScope", 0)
        if 128 <= tx:
            T.set_max_nreg(24, 0)
            if T.tl_shuffle_elect(128):
                T.ptx_arrive_barrier_expect_tx(T.get_mbarrier(8), 8192)
                T.tma_load(Q_desc, T.get_mbarrier(8), T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 16384, 4096, 2), 0, 0, bx * 64, 0, 0)
            for k in T.serial(4, annotations={"tl_pipeline_group": [[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]}):
                T.mbarrier_wait_parity(T.get_mbarrier(k % 2 + 4), T.bitwise_xor(k // 2, 1))
                if T.tl_shuffle_elect(128):
                    T.ptx_arrive_barrier_expect_tx(T.get_mbarrier(k % 2), 8192)
                    T.tma_load(K_desc, T.get_mbarrier(k % 2), T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, k % 2 * 4096, 4096, 2), 0, 0, k * 64, 0, 0)
                T.mbarrier_wait_parity(T.get_mbarrier(k % 2 + 6), T.bitwise_xor(k // 2, 1))
                if T.tl_shuffle_elect(128):
                    T.ptx_arrive_barrier_expect_tx(T.get_mbarrier(k % 2 + 2), 8192)
                    T.tma_load(V_desc, T.get_mbarrier(k % 2 + 2), T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 8192 + k % 2 * 4096, 4096, 2), 0, 0, k * 64, 0, 0)
        else:
            T.set_max_nreg(240, 1)
            for i in T.unroll(16):
                acc_o_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
            for i in T.unroll(2):
                logsum_1[i] = T.float32(0.0)
            for i in T.unroll(2):
                scores_max_1[i] = T.float32("-inf")
            T.fence_proxy_async()
            T.mbarrier_wait_parity(T.get_mbarrier(8), 0)
            for i in T.unroll(16):
                acc_s_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
            T.fence_proxy_async()
            T.mbarrier_wait_parity(T.get_mbarrier(0), 0)
            T.tl_gemm("tl::gemm_ss<64, 64, 64, 4, 1, 0, 1, 0, 64, 64, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 16384, 4096, 1), T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_s, 0, 4096, 3))
            T.ptx_arrive_barrier(T.get_mbarrier(4), 0, tx == 0)
            for i in T.unroll(2):
                scores_max_prev_1[i] = scores_max_1[i]
            for i in T.unroll(2):
                scores_max_1[i] = T.float32("-inf")
            for i in T.unroll(2):
                for rv in T.unroll(16):
                    scores_max_1[i] = T.max(scores_max_1[i], acc_s_1[rv % 8 * 4 + i * 2 + rv // 8])
                scores_max_1[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, 128>::run_hopper", scores_max_1[i])
            for i in T.unroll(2):
                scores_scale_1[i] = T.exp2(scores_max_prev_1[i] * T.float32(0.18033688) - scores_max_1[i] * T.float32(0.18033688))
            for i in T.unroll(32):
                acc_s_1[i] = T.exp2(acc_s_1[i] * T.float32(0.18033688) - scores_max_1[i % 4 // 2] * T.float32(0.18033688))
            for i in T.unroll(2):
                scores_sum_1[i] = T.float32(0.0)
                for rv in T.unroll(16):
                    scores_sum_1[i] = scores_sum_1[i] + acc_s_1[rv % 8 * 4 + i * 2 + rv // 8]
                scores_sum_1[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, 128>::run_hopper", scores_sum_1[i])
            for i in T.unroll(2):
                logsum_1[i] = logsum_1[i] * scores_scale_1[i] + scores_sum_1[i]
            for i in T.unroll(16):
                acc_s_cast_1[i * 2:i * 2 + 2] = T.Cast("float16x2", acc_s_1[i * 2:i * 2 + 2])
            for k in T.serial(3, annotations={"num_stages": 2, "tl_pipeline_order": [-1, 0, 3, 1, -1, 2], "tl_pipeline_stage": [-1, 0, 0, 1, -1, 1]}):
                for i in T.unroll(16):
                    acc_s_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
                T.fence_proxy_async()
                T.mbarrier_wait_parity(T.get_mbarrier((k + 1) % 2), (k + 1) // 2)
                T.tl_gemm("tl::gemm_ss<64, 64, 64, 4, 1, 0, 1, 0, 64, 64, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 16384, 4096, 1), T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, (k + 1) % 2 * 4096, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_s, 0, 4096, 3))
                T.ptx_arrive_barrier(T.get_mbarrier((k + 1) % 2 + 4), 0, tx == 0)
                for i in T.unroll(32):
                    acc_o_1[i] = acc_o_1[i] * scores_scale_1[i % 4 // 2]
                T.fence_proxy_async()
                T.mbarrier_wait_parity(T.get_mbarrier(k % 2 + 2), k // 2)
                T.tl_gemm("tl::gemm_rs<64, 64, 64, 4, 1, 0, 0, 0, 64, 64, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), acc_s_cast, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 8192 + k % 2 * 4096, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_o, 0, 4096, 3))
                T.ptx_arrive_barrier(T.get_mbarrier(k % 2 + 6), 0, tx == 0)
                for i in T.unroll(2):
                    scores_max_prev_1[i] = scores_max_1[i]
                for i in T.unroll(2):
                    scores_max_1[i] = T.float32("-inf")
                for i in T.unroll(2):
                    for rv in T.unroll(16):
                        scores_max_1[i] = T.max(scores_max_1[i], acc_s_1[rv % 8 * 4 + i * 2 + rv // 8])
                    scores_max_1[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, 128>::run_hopper", scores_max_1[i])
                for i in T.unroll(2):
                    scores_scale_1[i] = T.exp2(scores_max_prev_1[i] * T.float32(0.18033688) - scores_max_1[i] * T.float32(0.18033688))
                for i in T.unroll(32):
                    acc_s_1[i] = T.exp2(acc_s_1[i] * T.float32(0.18033688) - scores_max_1[i % 4 // 2] * T.float32(0.18033688))
                for i in T.unroll(2):
                    scores_sum_1[i] = T.float32(0.0)
                    for rv in T.unroll(16):
                        scores_sum_1[i] = scores_sum_1[i] + acc_s_1[rv % 8 * 4 + i * 2 + rv // 8]
                    scores_sum_1[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, 128>::run_hopper", scores_sum_1[i])
                for i in T.unroll(2):
                    logsum_1[i] = logsum_1[i] * scores_scale_1[i] + scores_sum_1[i]
                for i in T.unroll(16):
                    acc_s_cast_1[i * 2:i * 2 + 2] = T.Cast("float16x2", acc_s_1[i * 2:i * 2 + 2])
            for i in T.unroll(32):
                acc_o_1[i] = acc_o_1[i] * scores_scale_1[i % 4 // 2]
            T.fence_proxy_async()
            T.mbarrier_wait_parity(T.get_mbarrier(3), 1)
            T.tl_gemm("tl::gemm_rs<64, 64, 64, 4, 1, 0, 0, 0, 64, 64, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), acc_s_cast, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_o, 0, 4096, 3))
            T.ptx_arrive_barrier(T.get_mbarrier(7), 0, tx == 0)
            for i in T.unroll(32):
                acc_o_1[i] = acc_o_1[i] / logsum_1[i % 4 // 2]
            T.tvm_storage_sync("shared.dyn", 3, 128)
            for i in T.unroll(4):
                T.ptx_stmatrix(0, 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 16384 + (tx // 32 * 1024 + tx % 16 * 64 + i * 16 + tx % 32 // 16 * 8), 8, 2), T.pack_b16(T.Cast("float16", acc_o_1[i * 8]), T.Cast("float16", acc_o_1[i * 8 + 1])), T.pack_b16(T.Cast("float16", acc_o_1[i * 8 + 2]), T.Cast("float16", acc_o_1[i * 8 + 3])), T.pack_b16(T.Cast("float16", acc_o_1[i * 8 + 4]), T.Cast("float16", acc_o_1[i * 8 + 5])), T.pack_b16(T.Cast("float16", acc_o_1[i * 8 + 6]), T.Cast("float16", acc_o_1[i * 8 + 7])))
            T.fence_proxy_async()
            T.tvm_storage_sync("shared.dyn", 3, 128)
            if T.tl_shuffle_elect(128):
                T.tma_store(Output_desc, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 16384, 4096, 1), 0, 0, bx * 64, 0, 0)
                T.tma_store_arrive()
                T.tma_store_wait()