# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        Q = T.match_buffer(Q_handle, (8, 32, 4096, 128), "float16", strides=(16777216, 524288, 128, 1))
        K = T.match_buffer(K_handle, (8, 32, 4096, 128), "float16", strides=(16777216, 524288, 128, 1))
        V = T.match_buffer(V_handle, (8, 32, 4096, 128), "float16", strides=(16777216, 524288, 128, 1))
        Output = T.match_buffer(Output_handle, (8, 32, 4096, 128), "float16", strides=(16777216, 524288, 128, 1))
        with T.block("root"):
            T.reads(Q[0:8, 0:32, 0:4096, 0:128], K[0:8, 0:32, 0:4096, 0:128], V[0:8, 0:32, 0:4096, 0:128])
            T.writes(Output[0:8, 0:32, 0:4096, 0:128], Q[0:8, 0:32, 0:4096, 0:128], K[0:8, 0:32, 0:4096, 0:128], V[0:8, 0:32, 0:4096, 0:128])
            scores_max_prev = T.Buffer((2,), scope="local")
            scores_max = T.Buffer((2,), scope="local")
            scores_sum = T.Buffer((2,), scope="local")
            acc_s = T.Buffer((64,), scope="local")
            acc_s_cast = T.Buffer((64,), "float16", scope="local")
            scores_scale = T.Buffer((2,), scope="local")
            logsum = T.Buffer((2,), scope="local")
            acc_o = T.Buffer((64,), scope="local")
            T.block_attr({"layout_map": {scores_max_prev: metadata["tl.Fragment"][0], scores_max: metadata["tl.Fragment"][1], scores_sum: metadata["tl.Fragment"][2], acc_s: metadata["tl.Fragment"][3], acc_s_cast: metadata["tl.Fragment"][4], scores_scale: metadata["tl.Fragment"][5], logsum: metadata["tl.Fragment"][6], acc_o: metadata["tl.Fragment"][7]}})
            bx = T.launch_thread("blockIdx.x", 32)
            by = T.launch_thread("blockIdx.y", 32)
            bz = T.launch_thread("blockIdx.z", 8)
            tx = T.launch_thread("threadIdx.x", 384)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.block("tilelang_root"):
                Q_shared = T.Buffer((2, 16, 512), "float16", scope="shared.dyn")
                K_shared = T.Buffer((2, 2, 16, 512), "float16", scope="shared.dyn")
                V_shared = T.Buffer((2, 2, 16, 512), "float16", scope="shared.dyn")
                O_shared = T.Buffer((128, 128), "float16", scope="shared.dyn")
                T.reads(Q_shared[0:2, 0:16, 0:512], K_shared[0:2, 0:2, 0:16, 0:512], scores_max[0:2], scores_max_prev[0:2], scores_sum[0:2], logsum[0:2], scores_scale[0:2], acc_s_cast[0:64], V_shared[0:2, 0:2, 0:16, 0:512], O_shared[0:128, 0:128], Q[0:8, 0:32, 0:4096, 0:128], K[0:8, 0:32, 0:4096, 0:128], V[0:8, 0:32, 0:4096, 0:128], acc_s[0:64], acc_o[0:64])
                T.writes(Q_shared[0:2, 0:16, 0:512], K_shared[0:2, 0:2, 0:16, 0:512], V_shared[0:2, 0:2, 0:16, 0:512], logsum[0:2], scores_max[0:2], scores_max_prev[0:2], scores_scale[0:2], scores_sum[0:2], acc_s_cast[0:64], O_shared[0:128, 0:128], Output[0:8, 0:32, 0:4096, 0:128], Q[0:8, 0:32, 0:4096, 0:128], K[0:8, 0:32, 0:4096, 0:128], V[0:8, 0:32, 0:4096, 0:128], acc_s[0:64], acc_o[0:64])
                T.block_attr({"layout_map": {scores_max_prev: metadata["tl.Fragment"][0], scores_max: metadata["tl.Fragment"][1], scores_sum: metadata["tl.Fragment"][2], acc_s: metadata["tl.Fragment"][3], acc_s_cast: metadata["tl.Fragment"][4], scores_scale: metadata["tl.Fragment"][5], logsum: metadata["tl.Fragment"][6], acc_o: metadata["tl.Fragment"][7]}})
                Q_shared = T.alloc_buffer((2, 16, 512), "float16", data=Q_shared.data, scope="shared.dyn")
                K_shared = T.alloc_buffer((2, 2, 16, 512), "float16", data=K_shared.data, scope="shared.dyn")
                V_shared = T.alloc_buffer((2, 2, 16, 512), "float16", data=V_shared.data, scope="shared.dyn")
                O_shared = T.alloc_buffer((128, 128), "float16", data=O_shared.data, scope="shared.dyn")
                acc_s = T.alloc_buffer((64,), data=acc_s.data, scope="local")
                acc_s_cast = T.alloc_buffer((64,), "float16", data=acc_s_cast.data, scope="local")
                acc_o = T.alloc_buffer((64,), data=acc_o.data, scope="local")
                scores_max = T.alloc_buffer((2,), data=scores_max.data, scope="local")
                scores_max_prev = T.alloc_buffer((2,), data=scores_max_prev.data, scope="local")
                scores_scale = T.alloc_buffer((2,), data=scores_scale.data, scope="local")
                scores_sum = T.alloc_buffer((2,), data=scores_sum.data, scope="local")
                logsum = T.alloc_buffer((2,), data=logsum.data, scope="local")
                T.create_list_of_mbarrier(1, 1, 1, 1, 256, 256, 256, 256, 1)
                T.attr([128, 256], "kWarpSpecializationScope", 0)
                if tx >= 256:
                    with T.block("", no_realize=True):
                        T.reads(Q[0:8, 0:32, 0:4096, 0:128])
                        T.writes(Q_shared[0:2, 0:16, 0:512], Q[0:8, 0:32, 0:4096, 0:128])
                        T.block_attr({"stmt_group": 1})
                        if T.tl_shuffle_elect(128):
                            T.ptx_arrive_barrier_expect_tx(T.get_mbarrier(8), 32768)
                            for i in T.unroll(2):
                                T.tma_load(T.create_tma_descriptor(6, 4, Q.data, 128, 4096, 32, 8, T.int64(2), T.int64(256), T.int64(1048576), T.int64(33554432), 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(8), T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, i * 8192, 8192, 2), i * 64, bx * 128, by, bz, 0)
                    with T.block("", no_realize=True):
                        T.reads(K[0:8, 0:32, 0:4096, 0:128], V[0:8, 0:32, 0:4096, 0:128])
                        T.writes(K_shared[0:2, 0:2, 0:16, 0:512], V_shared[0:2, 0:2, 0:16, 0:512], K[0:8, 0:32, 0:4096, 0:128], V[0:8, 0:32, 0:4096, 0:128])
                        T.block_attr({"stmt_group": 1})
                        for k in T.serial(32, annotations={"tl_pipeline_group": [[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]}):
                            with T.block("", no_realize=True):
                                T.reads(K[0:8, 0:32, 0:4096, 0:128])
                                T.writes(K_shared[0:2, 0:2, 0:16, 0:512], K[0:8, 0:32, 0:4096, 0:128])
                                T.block_attr({"stmt_group": 1})
                                T.mbarrier_wait_parity(T.get_mbarrier(k % 2 + 4), T.bitwise_xor(k // 2 % 2, 1))
                                if T.tl_shuffle_elect(128):
                                    T.ptx_arrive_barrier_expect_tx(T.get_mbarrier(k % 2), 32768)
                                    for i in T.unroll(2):
                                        T.tma_load(T.create_tma_descriptor(6, 4, K.data, 128, 4096, 32, 8, T.int64(2), T.int64(256), T.int64(1048576), T.int64(33554432), 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(k % 2), T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, i * 8192 + k % 2 * 16384, 8192, 2), i * 64, k * 128, by, bz, 0)
                            with T.block("", no_realize=True):
                                T.reads(V[0:8, 0:32, 0:4096, 0:128])
                                T.writes(V_shared[0:2, 0:2, 0:16, 0:512], V[0:8, 0:32, 0:4096, 0:128])
                                T.block_attr({"stmt_group": 1})
                                T.mbarrier_wait_parity(T.get_mbarrier(k % 2 + 6), T.bitwise_xor(k // 2 % 2, 1))
                                if T.tl_shuffle_elect(128):
                                    T.ptx_arrive_barrier_expect_tx(T.get_mbarrier(k % 2 + 2), 32768)
                                    for i in T.unroll(2):
                                        T.tma_load(T.create_tma_descriptor(6, 4, V.data, 128, 4096, 32, 8, T.int64(2), T.int64(256), T.int64(1048576), T.int64(33554432), 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), T.get_mbarrier(k % 2 + 2), T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, i * 8192 + k % 2 * 16384, 8192, 2), i * 64, k * 128, by, bz, 0)
                else:
                    with T.block("", no_realize=True):
                        T.reads()
                        T.writes(acc_o[0:64])
                        T.block_attr({"stmt_group": 1})
                        for i in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                            for vec in T.vectorized(2):
                                acc_o[i * 2 + vec] = T.float32(0.0)
                    with T.block("", no_realize=True):
                        T.reads()
                        T.writes(logsum[0:2])
                        T.block_attr({"stmt_group": 1})
                        for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                            logsum[i] = T.float32(0.0)
                    with T.block("", no_realize=True):
                        T.reads()
                        T.writes(scores_max[0:2])
                        T.block_attr({"stmt_group": 1})
                        for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                            scores_max[i] = T.infinity("float") * T.float32(-1.0)
                    with T.block("", no_realize=True):
                        T.reads(Q_shared[0:2, 0:16, 0:512], K_shared[0:2, 0:2, 0:16, 0:512], scores_max[0:2], scores_max_prev[0:2], scores_sum[0:2], logsum[0:2], scores_scale[0:2], acc_s_cast[0:64], V_shared[0:2, 0:2, 0:16, 0:512], acc_s[0:64], acc_o[0:64])
                        T.writes(scores_max_prev[0:2], scores_max[0:2], scores_scale[0:2], scores_sum[0:2], logsum[0:2], acc_s_cast[0:64], acc_s[0:64], acc_o[0:64])
                        T.block_attr({"stmt_group": 1})
                        T.mbarrier_wait_parity(T.get_mbarrier(8), 0)
                        with T.block(""):
                            T.reads(Q_shared[0:2, 0:16, 0:512], K_shared[0:2, 0:2, 0:16, 0:512], scores_max[0:2], scores_max_prev[0:2], scores_sum[0:2], logsum[0:2], scores_scale[0:2], acc_s_cast[0:64], V_shared[0:2, 0:2, 0:16, 0:512], acc_s[0:64], acc_o[0:64])
                            T.writes(scores_max_prev[0:2], scores_max[0:2], scores_scale[0:2], scores_sum[0:2], logsum[0:2], acc_s_cast[0:64], acc_s[0:64], acc_o[0:64])
                            with T.block(""):
                                T.reads(Q_shared[0:2, 0:16, 0:512], K_shared[0:2, 0:2, 0:16, 0:512], scores_max[0:2], scores_max_prev[0:2], scores_sum[0:2], logsum[0:2], scores_scale[0:2], acc_s[0:64])
                                T.writes(scores_max_prev[0:2], scores_max[0:2], scores_scale[0:2], scores_sum[0:2], logsum[0:2], acc_s_cast[0:64], acc_s[0:64])
                                with T.block(""):
                                    T.reads(Q_shared[0:2, 0:16, 0:512], K_shared[0:2, 0:2, 0:16, 0:512], acc_s[0:64])
                                    T.writes(acc_s[0:64])
                                    with T.block("", no_realize=True):
                                        T.reads(Q_shared[0:2, 0:16, 0:512], K_shared[0:2, 0:2, 0:16, 0:512], acc_s[0:64])
                                        T.writes(acc_s[0:64])
                                        T.block_attr({"stmt_group": 1})
                                        for i in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                            for vec in T.vectorized(2):
                                                acc_s[i * 2 + vec] = T.float32(0.0)
                                        T.mbarrier_wait_parity(T.get_mbarrier(T.FloorMod(0, 2)), T.FloorDiv(0, 2) % 2)
                                        T.tl_gemm("tl::gemm_ss<128, 128, 128, 8, 1, 0, 1, 0, 128, 128, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, T.FloorMod(0, 2) * 16384, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_s.data, 0, 16384, 3))
                                        T.ptx_arrive_barrier(T.get_mbarrier(T.FloorMod(0, 2) + 4))
                                with T.block(""):
                                    T.reads(scores_max[0:2], acc_s[0:64], scores_max_prev[0:2], scores_sum[0:2], logsum[0:2], scores_scale[0:2])
                                    T.writes(scores_max_prev[0:2], scores_max[0:2], scores_scale[0:2], acc_s[0:64], scores_sum[0:2], logsum[0:2], acc_s_cast[0:64])
                                    with T.block("", no_realize=True):
                                        T.reads(scores_max[0:2], acc_s[0:64], scores_max_prev[0:2], scores_sum[0:2], logsum[0:2], scores_scale[0:2])
                                        T.writes(scores_max_prev[0:2], scores_max[0:2], scores_scale[0:2], acc_s[0:64], scores_sum[0:2], logsum[0:2], acc_s_cast[0:64])
                                        T.block_attr({"stmt_group": 1})
                                        for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                            scores_max_prev[i] = scores_max[i]
                                        for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                            scores_max[i] = T.infinity("float") * T.float32(-1.0)
                                        for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                            for rv in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                scores_max[i] = T.max(scores_max[i], acc_s[rv % 16 * 4 + i * 2 + rv // 16])
                                            scores_max[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, 256>::run_hopper", scores_max[i])
                                        for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                            scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.1275174307460247) - scores_max[i] * T.float32(0.1275174307460247))
                                        for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                            acc_s[i] = T.exp2(acc_s[i] * T.float32(0.1275174307460247) - scores_max[i % 4 // 2] * T.float32(0.1275174307460247))
                                        for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                            scores_sum[i] = T.float32(0.0)
                                            for rv in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                scores_sum[i] = scores_sum[i] + acc_s[rv % 16 * 4 + i * 2 + rv // 16]
                                            scores_sum[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, 256>::run_hopper", scores_sum[i])
                                        for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                                        for i in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                            for vec in T.vectorized(2):
                                                acc_s_cast[i * 2 + vec] = T.Cast("float16", acc_s[i * 2 + vec])
                            with T.block(""):
                                T.reads(Q_shared[0:2, 0:16, 0:512], K_shared[0:2, 0:2, 0:16, 0:512], scores_scale[0:2], acc_s_cast[0:64], V_shared[0:2, 0:2, 0:16, 0:512], scores_max[0:2], scores_max_prev[0:2], scores_sum[0:2], logsum[0:2], acc_s[0:64], acc_o[0:64])
                                T.writes(scores_max_prev[0:2], scores_max[0:2], scores_scale[0:2], scores_sum[0:2], logsum[0:2], acc_s_cast[0:64], acc_s[0:64], acc_o[0:64])
                                for k in T.serial(31, annotations={"num_stages": 2, "tl_pipeline_order": [-1, 0, 3, 1, -1, 2], "tl_pipeline_stage": [-1, 0, 0, 1, -1, 1]}):
                                    with T.block(""):
                                        T.reads(Q_shared[0:2, 0:16, 0:512], K_shared[0:2, 0:2, 0:16, 0:512], acc_s[0:64])
                                        T.writes(acc_s[0:64])
                                        with T.block("", no_realize=True):
                                            T.reads(Q_shared[0:2, 0:16, 0:512], K_shared[0:2, 0:2, 0:16, 0:512], acc_s[0:64])
                                            T.writes(acc_s[0:64])
                                            T.block_attr({"stmt_group": 1})
                                            for i in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                for vec in T.vectorized(2):
                                                    acc_s[i * 2 + vec] = T.float32(0.0)
                                            T.mbarrier_wait_parity(T.get_mbarrier((k + 1) % 2), (k + 1) // 2 % 2)
                                            T.tl_gemm("tl::gemm_ss<128, 128, 128, 8, 1, 0, 1, 0, 128, 128, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, (k + 1) % 2 * 16384, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_s.data, 0, 16384, 3))
                                            T.ptx_arrive_barrier(T.get_mbarrier((k + 1) % 2 + 4))
                                    with T.block(""):
                                        T.reads(acc_o[0:64], scores_scale[0:2])
                                        T.writes(acc_o[0:64])
                                        with T.block("", no_realize=True):
                                            T.reads(acc_o[0:64], scores_scale[0:2])
                                            T.writes(acc_o[0:64])
                                            T.block_attr({"stmt_group": 1})
                                            for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                acc_o[i] = acc_o[i] * scores_scale[i % 4 // 2]
                                    with T.block(""):
                                        T.reads(acc_s_cast[0:64], V_shared[0:2, 0:2, 0:16, 0:512], acc_o[0:64])
                                        T.writes(acc_o[0:64])
                                        with T.block("", no_realize=True):
                                            T.reads(acc_s_cast[0:64], V_shared[0:2, 0:2, 0:16, 0:512], acc_o[0:64])
                                            T.writes(acc_o[0:64])
                                            T.block_attr({"stmt_group": 1})
                                            T.mbarrier_wait_parity(T.get_mbarrier((k - 1 + 1) % 2 + 2), (k - 1 + 1) // 2 % 2)
                                            T.tl_gemm("tl::gemm_rs<128, 128, 128, 8, 1, 0, 0, 0, 128, 128, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), acc_s_cast.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, (k - 1 + 1) % 2 * 16384, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_o.data, 0, 16384, 3))
                                            T.ptx_arrive_barrier(T.get_mbarrier((k - 1 + 1) % 2 + 6))
                                    with T.block(""):
                                        T.reads(scores_max[0:2], acc_s[0:64], scores_max_prev[0:2], scores_sum[0:2], logsum[0:2], scores_scale[0:2])
                                        T.writes(scores_max_prev[0:2], scores_max[0:2], scores_scale[0:2], acc_s[0:64], scores_sum[0:2], logsum[0:2], acc_s_cast[0:64])
                                        with T.block("", no_realize=True):
                                            T.reads(scores_max[0:2], acc_s[0:64], scores_max_prev[0:2], scores_sum[0:2], logsum[0:2], scores_scale[0:2])
                                            T.writes(scores_max_prev[0:2], scores_max[0:2], scores_scale[0:2], acc_s[0:64], scores_sum[0:2], logsum[0:2], acc_s_cast[0:64])
                                            T.block_attr({"stmt_group": 1})
                                            for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                scores_max_prev[i] = scores_max[i]
                                            for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                scores_max[i] = T.infinity("float") * T.float32(-1.0)
                                            for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                for rv in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                    scores_max[i] = T.max(scores_max[i], acc_s[rv % 16 * 4 + i * 2 + rv // 16])
                                                scores_max[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, 256>::run_hopper", scores_max[i])
                                            for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.1275174307460247) - scores_max[i] * T.float32(0.1275174307460247))
                                            for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                acc_s[i] = T.exp2(acc_s[i] * T.float32(0.1275174307460247) - scores_max[i % 4 // 2] * T.float32(0.1275174307460247))
                                            for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                scores_sum[i] = T.float32(0.0)
                                                for rv in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                    scores_sum[i] = scores_sum[i] + acc_s[rv % 16 * 4 + i * 2 + rv // 16]
                                                scores_sum[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, 256>::run_hopper", scores_sum[i])
                                            for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                                            for i in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                                for vec in T.vectorized(2):
                                                    acc_s_cast[i * 2 + vec] = T.Cast("float16", acc_s[i * 2 + vec])
                            with T.block(""):
                                T.reads(scores_scale[0:2], acc_s_cast[0:64], V_shared[0:2, 0:2, 0:16, 0:512], acc_o[0:64])
                                T.writes(acc_o[0:64])
                                with T.block(""):
                                    T.reads(acc_o[0:64], scores_scale[0:2])
                                    T.writes(acc_o[0:64])
                                    with T.block("", no_realize=True):
                                        T.reads(acc_o[0:64], scores_scale[0:2])
                                        T.writes(acc_o[0:64])
                                        T.block_attr({"stmt_group": 1})
                                        for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                            acc_o[i] = acc_o[i] * scores_scale[i % 4 // 2]
                                with T.block(""):
                                    T.reads(acc_s_cast[0:64], V_shared[0:2, 0:2, 0:16, 0:512], acc_o[0:64])
                                    T.writes(acc_o[0:64])
                                    with T.block("", no_realize=True):
                                        T.reads(acc_s_cast[0:64], V_shared[0:2, 0:2, 0:16, 0:512], acc_o[0:64])
                                        T.writes(acc_o[0:64])
                                        T.block_attr({"stmt_group": 1})
                                        T.mbarrier_wait_parity(T.get_mbarrier(T.FloorMod(31, 2) + 2), T.FloorDiv(31, 2) % 2)
                                        T.tl_gemm("tl::gemm_rs<128, 128, 128, 8, 1, 0, 0, 0, 128, 128, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), acc_s_cast.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, T.FloorMod(31, 2) * 16384, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_o.data, 0, 16384, 3))
                                        T.ptx_arrive_barrier(T.get_mbarrier(T.FloorMod(31, 2) + 6))
                    with T.block("", no_realize=True):
                        T.reads(acc_o[0:64], logsum[0:2])
                        T.writes(acc_o[0:64])
                        T.block_attr({"stmt_group": 1})
                        for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                            acc_o[i] = acc_o[i] / logsum[i % 4 // 2]
                    with T.block("", no_realize=True):
                        T.reads(acc_o[0:64])
                        T.writes(O_shared[0:128, 0:128])
                        T.block_attr({"stmt_group": 1})
                        for i in T.unroll(8, annotations={"pragma_unroll_explicit": T.bool(False)}):
                            T.ptx_stmatrix(0, 4, T.tvm_access_ptr(T.type_annotation("float16"), O_shared.data, tx // 32 * 2048 + tx % 16 * 128 + i * 16 + tx % 32 // 16 * 8, 8, 2), T.pack_b16(T.Cast("float16", acc_o[i * 8]), T.Cast("float16", acc_o[i * 8 + 1])), T.pack_b16(T.Cast("float16", acc_o[i * 8 + 2]), T.Cast("float16", acc_o[i * 8 + 3])), T.pack_b16(T.Cast("float16", acc_o[i * 8 + 4]), T.Cast("float16", acc_o[i * 8 + 5])), T.pack_b16(T.Cast("float16", acc_o[i * 8 + 6]), T.Cast("float16", acc_o[i * 8 + 7])))
                    with T.block("", no_realize=True):
                        T.reads(O_shared[0:128, 0:128])
                        T.writes(Output[0:8, 0:32, 0:4096, 0:128])
                        T.block_attr({"stmt_group": 1})
                        if T.tl_shuffle_elect(256):
                            T.tma_store(T.tvm_access_ptr(T.type_annotation("float16"), Output.data, bz * 16777216 + by * 524288 + bx * 16384, 16384, 2), T.tvm_access_ptr(T.type_annotation("float16"), O_shared.data, 0, 16384, 1), 32768, 0, 0)
