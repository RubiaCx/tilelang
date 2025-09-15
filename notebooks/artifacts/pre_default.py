# from tvm.script import tir as T

@T.prim_func
def kernel_impl(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle):
    Q = T.match_buffer(Q_handle, (1, 256, 1, 64), "float16", strides=(16384, 64, 64, 1))
    K = T.match_buffer(K_handle, (1, 256, 1, 64), "float16", strides=(16384, 64, 64, 1))
    V = T.match_buffer(V_handle, (1, 256, 1, 64), "float16", strides=(16384, 64, 64, 1))
    Output = T.match_buffer(Output_handle, (1, 256, 1, 64), "float16", strides=(16384, 64, 64, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 4)
    by = T.launch_thread("blockIdx.y", 1)
    bz = T.launch_thread("blockIdx.z", 1)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        loop_range = T.int32()
        T.reads(Q[bz, bx * 64, by, 0], K[bz, 0:loop_range * 64 - 63, by, 0], V[bz, 0:loop_range * 64 - 63, by, 0], Output[bz, bx * 64, by, 0])
        T.writes()
        Q_shared = T.alloc_buffer((64, 64), "float16", scope="shared.dyn")
        K_shared = T.alloc_buffer((64, 64), "float16", scope="shared.dyn")
        V_shared = T.alloc_buffer((64, 64), "float16", scope="shared.dyn")
        O_shared = T.alloc_buffer((64, 64), "float16", scope="shared.dyn")
        acc_s = T.alloc_buffer((64, 64), scope="local.fragment")
        acc_s_cast = T.alloc_buffer((64, 64), "float16", scope="local.fragment")
        acc_o = T.alloc_buffer((64, 64), scope="local.fragment")
        scores_max = T.alloc_buffer((64,), scope="local.fragment")
        scores_max_prev = T.alloc_buffer((64,), scope="local.fragment")
        scores_scale = T.alloc_buffer((64,), scope="local.fragment")
        scores_sum = T.alloc_buffer((64,), scope="local.fragment")
        logsum = T.alloc_buffer((64,), scope="local.fragment")
        T.copy(T.region(Q[bz, bx * 64, by, 0], 1, 1, 64, 1, 64), T.region(Q_shared[0, 0], 2, 64, 64), -1, T.bool(False), 0)
        T.fill(T.tvm_access_ptr(T.type_annotation("float32"), acc_o.data, 0, 4096, 2), 0)
        T.fill(T.tvm_access_ptr(T.type_annotation("float32"), logsum.data, 0, 64, 2), 0)
        T.fill(T.tvm_access_ptr(T.type_annotation("float32"), scores_max.data, 0, 64, 2), T.float32("-inf"))
        with T.LetStmt(4, var=loop_range):
            for k in T.serial(loop_range, annotations={"num_stages": 2}):
                T.copy(T.region(K[bz, k * 64, by, 0], 1, 1, 64, 1, 64), T.region(K_shared[0, 0], 2, 64, 64), -1, T.bool(False), 0)
                T.fill(T.tvm_access_ptr(T.type_annotation("float32"), acc_s.data, 0, 4096, 2), 0)
                T.gemm(T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_s.data, 0, 4096, 3), T.bool(False), T.bool(True), 64, 64, 64, 1, T.bool(False), 64, 64, 0, 0, 1, 0)
                T.copy(T.region(scores_max[0], 1, 64), T.region(scores_max_prev[0], 2, 64), -1, T.bool(False), 0)
                T.fill(T.tvm_access_ptr(T.type_annotation("float32"), scores_max.data, 0, 64, 2), T.float32("-inf"))
                T.reduce(T.tvm_access_ptr(T.type_annotation("float32"), acc_s.data, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), scores_max.data, 0, 64, 2), "max", 1, T.bool(False))
                for i in T.parallel(64):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                T.reduce(T.tvm_access_ptr(T.type_annotation("float32"), acc_s.data, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), scores_sum.data, 0, 64, 2), "sum", 1, T.bool(True))
                for i in T.parallel(64):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(T.region(acc_s[0, 0], 1, 64, 64), T.region(acc_s_cast[0, 0], 2, 64, 64), -1, T.bool(False), 0)
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                T.copy(T.region(V[bz, k * 64, by, 0], 1, 1, 64, 1, 64), T.region(V_shared[0, 0], 2, 64, 64), -1, T.bool(False), 0)
                T.gemm(T.tvm_access_ptr(T.type_annotation("float16"), acc_s_cast.data, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), acc_o.data, 0, 4096, 3), T.bool(False), T.bool(False), 64, 64, 64, 1, T.bool(False), 64, 64, 0, 0, 1, 0)
            for i in T.parallel(64):
                for j in T.parallel(64):
                    acc_o[i, j] = acc_o[i, j] / logsum[i]
            T.copy(T.region(acc_o[0, 0], 1, 64, 64), T.region(O_shared[0, 0], 2, 64, 64), -1, T.bool(False), 0)
            T.copy(T.region(O_shared[0, 0], 1, 64, 64), T.region(Output[bz, bx * 64, by, 0], 2, 1, 64, 1, 64), -1, T.bool(False), 0)