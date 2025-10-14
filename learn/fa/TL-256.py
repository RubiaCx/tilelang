# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        Q = T.match_buffer(Q_handle, (4, 32, 256, 128), "float16", strides=(1048576, 32768, 128, 1))
        K = T.match_buffer(K_handle, (4, 32, 256, 128), "float16", strides=(1048576, 32768, 128, 1))
        V = T.match_buffer(V_handle, (4, 32, 256, 128), "float16", strides=(1048576, 32768, 128, 1))
        Output = T.match_buffer(Output_handle, (4, 32, 256, 128), "float16", strides=(1048576, 32768, 128, 1))
        bx = T.launch_thread("blockIdx.x", 2)
        by = T.launch_thread("blockIdx.y", 32)
        bz = T.launch_thread("blockIdx.z", 4)
        tx = T.launch_thread("threadIdx.x", 384)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        Q_shared = T.decl_buffer((2, 16, 512), "float16", scope="shared.dyn")
        K_shared = T.decl_buffer((2, 2, 16, 512), "float16", scope="shared.dyn")
        V_shared = T.decl_buffer((2, 2, 16, 512), "float16", scope="shared.dyn")
        O_shared = T.decl_buffer((128, 128), "float16", scope="shared.dyn")
        acc_s = T.decl_buffer((64,), scope="local")
        acc_s_cast = T.decl_buffer((64,), "float16", scope="local")
        acc_o = T.decl_buffer((64,), scope="local")
        scores_max = T.decl_buffer((2,), scope="local")
        scores_max_prev = T.decl_buffer((2,), scope="local")
        scores_scale = T.decl_buffer((2,), scope="local")
        scores_sum = T.decl_buffer((2,), scope="local")
        logsum = T.decl_buffer((2,), scope="local")
        T.create_list_of_mbarrier(1, 1, 1, 1, 256, 256, 256, 256, 1)
        T.attr([128, 256], "kWarpSpecializationScope", 0)
        if tx >= 256:
            if T.tl_shuffle_elect(128):
                T.ptx_arrive_barrier_expect_tx(T.get_mbarrier(8), 32768)
                for i in T.unroll(2):
                    T.tma_load(T.create_tma_descriptor(6, 4, Q.data, 128, 256, 32, 4, T.int64(2), T.int64(256), T.int64(65536), T.int64(2097152), 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(8), T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, i * 8192, 8192, 2), i * 64, bx * 128, by, bz, 0)
            for k in range(2):
                T.mbarrier_wait_parity(T.get_mbarrier(k % 2 + 4), T.bitwise_xor(k // 2 % 2, 1))
                if T.tl_shuffle_elect(128):
                    T.ptx_arrive_barrier_expect_tx(T.get_mbarrier(k % 2), 32768)
                    for i in T.unroll(2):
                        T.tma_load(T.create_tma_descriptor(6, 4, K.data, 128, 256, 32, 4, T.int64(2), T.int64(256), T.int64(65536), T.int64(2097152), 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(k % 2), T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, i * 8192 + k % 2 * 16384, 8192, 2), i * 64, k * 128, by, bz, 0)
                T.mbarrier_wait_parity(T.get_mbarrier(k % 2 + 6), T.bitwise_xor(k // 2 % 2, 1))
                if T.tl_shuffle_elect(128):
                    T.ptx_arrive_barrier_expect_tx(T.get_mbarrier(k % 2 + 2), 32768)
                    for i in T.unroll(2):
                        T.tma_load(T.create_tma_descriptor(6, 4, V.data, 128, 256, 32, 4, T.int64(2), T.int64(256), T.int64(65536), T.int64(2097152), 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(k % 2 + 2), T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, i * 8192 + k % 2 * 16384, 8192, 2), i * 64, k * 128, by, bz, 0)
        else:
            i = T.int32()
            with T.attr(i, "pragma_unroll_explicit", T.bool(False)):
                for i in T.unroll(32):
                    for vec in T.vectorized(2):
                        acc_o[i * 2 + vec] = T.float32(0.0)
            i_1 = T.int32()
            with T.attr(i_1, "pragma_unroll_explicit", T.bool(False)):
                for i_1 in T.unroll(2):
                    logsum[i_1] = T.float32(0.0)
            i_2 = T.int32()
            with T.attr(i_2, "pragma_unroll_explicit", T.bool(False)):
                for i_2 in T.unroll(2):
                    scores_max[i_2] = T.float32("-inf")
            T.mbarrier_wait_parity(T.get_mbarrier(8), 0)
            for k in range(2):
                i_3 = T.int32()
                with T.attr(i_3, "pragma_unroll_explicit", T.bool(False)):
                    for i_3 in T.unroll(32):
                        for vec in T.vectorized(2):
                            acc_s[i_3 * 2 + vec] = T.float32(0.0)
                T.mbarrier_wait_parity(T.get_mbarrier(k % 2), k // 2 % 2)
                T.tl_gemm("tl::gemm_ss<128, 128, 128, 8, 1, 0, 1, 0, 128, 128, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, k % 2 * 16384, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_s.data, 0, 16384, 3))
                T.ptx_arrive_barrier(T.get_mbarrier(k % 2 + 4))
                i_4 = T.int32()
                with T.attr(i_4, "pragma_unroll_explicit", T.bool(False)):
                    for i_4 in T.unroll(2):
                        scores_max_prev[i_4] = scores_max[i_4]
                i_5 = T.int32()
                with T.attr(i_5, "pragma_unroll_explicit", T.bool(False)):
                    for i_5 in T.unroll(2):
                        scores_max[i_5] = T.float32("-inf")
                i_6 = T.int32()
                with T.attr(i_6, "pragma_unroll_explicit", T.bool(False)):
                    for i_6 in T.unroll(2):
                        rv = T.int32()
                        with T.attr(rv, "pragma_unroll_explicit", T.bool(False)):
                            for rv in T.unroll(32):
                                scores_max[i_6] = T.max(scores_max[i_6], acc_s[rv % 16 * 4 + i_6 * 2 + rv // 16])
                        scores_max[i_6] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, 256>::run_hopper", scores_max[i_6])
                i_7 = T.int32()
                with T.attr(i_7, "pragma_unroll_explicit", T.bool(False)):
                    for i_7 in T.unroll(2):
                        scores_scale[i_7] = T.exp2(scores_max_prev[i_7] * T.float32(0.1275174307460247) - scores_max[i_7] * T.float32(0.1275174307460247))
                i_8 = T.int32()
                with T.attr(i_8, "pragma_unroll_explicit", T.bool(False)):
                    for i_8 in T.unroll(64):
                        acc_s[i_8] = T.exp2(acc_s[i_8] * T.float32(0.1275174307460247) - scores_max[i_8 % 4 // 2] * T.float32(0.1275174307460247))
                i_9 = T.int32()
                with T.attr(i_9, "pragma_unroll_explicit", T.bool(False)):
                    for i_9 in T.unroll(2):
                        scores_sum[i_9] = T.float32(0.0)
                        rv = T.int32()
                        with T.attr(rv, "pragma_unroll_explicit", T.bool(False)):
                            for rv in T.unroll(32):
                                scores_sum[i_9] = scores_sum[i_9] + acc_s[rv % 16 * 4 + i_9 * 2 + rv // 16]
                        scores_sum[i_9] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, 256>::run_hopper", scores_sum[i_9])
                i_10 = T.int32()
                with T.attr(i_10, "pragma_unroll_explicit", T.bool(False)):
                    for i_10 in T.unroll(2):
                        logsum[i_10] = logsum[i_10] * scores_scale[i_10] + scores_sum[i_10]
                i_11 = T.int32()
                with T.attr(i_11, "pragma_unroll_explicit", T.bool(False)):
                    for i_11 in T.unroll(32):
                        for vec in T.vectorized(2):
                            acc_s_cast[i_11 * 2 + vec] = T.Cast("float16", acc_s[i_11 * 2 + vec])
                i_12 = T.int32()
                with T.attr(i_12, "pragma_unroll_explicit", T.bool(False)):
                    for i_12 in T.unroll(64):
                        acc_o[i_12] = acc_o[i_12] * scores_scale[i_12 % 4 // 2]
                T.mbarrier_wait_parity(T.get_mbarrier(k % 2 + 2), k // 2 % 2)
                T.tl_gemm("tl::gemm_rs<128, 128, 128, 8, 1, 0, 0, 0, 128, 128, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), acc_s_cast.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, k % 2 * 16384, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_o.data, 0, 16384, 3))
                T.ptx_arrive_barrier(T.get_mbarrier(k % 2 + 6))
            i_3 = T.int32()
            with T.attr(i_3, "pragma_unroll_explicit", T.bool(False)):
                for i_3 in T.unroll(64):
                    acc_o[i_3] = acc_o[i_3] / logsum[i_3 % 4 // 2]
            i_4 = T.int32()
            with T.attr(i_4, "pragma_unroll_explicit", T.bool(False)):
                for i_4 in T.unroll(8):
                    T.ptx_stmatrix(0, 4, T.tvm_access_ptr(T.type_annotation("float16"), O_shared.data, tx // 32 * 2048 + tx % 16 * 128 + i_4 * 16 + tx % 32 // 16 * 8, 8, 2), T.pack_b16(T.Cast("float16", acc_o[i_4 * 8]), T.Cast("float16", acc_o[i_4 * 8 + 1])), T.pack_b16(T.Cast("float16", acc_o[i_4 * 8 + 2]), T.Cast("float16", acc_o[i_4 * 8 + 3])), T.pack_b16(T.Cast("float16", acc_o[i_4 * 8 + 4]), T.Cast("float16", acc_o[i_4 * 8 + 5])), T.pack_b16(T.Cast("float16", acc_o[i_4 * 8 + 6]), T.Cast("float16", acc_o[i_4 * 8 + 7])))
            if T.tl_shuffle_elect(256):
                T.tma_store(T.tvm_access_ptr(T.type_annotation("float16"), Output.data, bz * 1048576 + by * 32768 + bx * 16384, 16384, 2), T.tvm_access_ptr(T.type_annotation("float16"), O_shared.data, 0, 16384, 1), 32768, 0)
2025-09-17 14:12:12  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`                                                                                                                                  
Compiling configurations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:11<00:00, 11.56s/it]
Tuned Latency 0.046112000942230225 with config {'block_M': 128, 'block_N': 128, 'num_stages': 2, 'threads': 256} at index 0                                                                                                            
Bench configurations: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 26.36it/s, best_latency=0.0461]
Best latency: 0.046112000942230225
Best TFlops: 93.14207165680789
Best config: {'block_M': 128, 'block_N': 128, 'num_stages': 2, 'threads': 256}
Ref latency: None