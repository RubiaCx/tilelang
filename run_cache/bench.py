#!/usr/bin/env python3
import argparse
import torch
try:
    from .runner import KernelLib, parse_wrapped_kernel_params
except ImportError:
    # 允许脚本直接执行: 将项目根目录加入 sys.path 后使用绝对导入
    import sys, os
    # __file__ = /home/chenxi/tilelang/cache/bench.py
    # 需要加入 /home/chenxi (项目根)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from tilelang.cache.runner import KernelLib, parse_wrapped_kernel_params


def _measure_latency_ms(k: KernelLib, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor, warmup: int, repeat: int) -> float:
    for _ in range(max(0, warmup)):
        k.call(Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), 0)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(max(1, repeat)):
        k.call(Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), 0)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return total_ms / max(1, repeat)


def main():
    parser = argparse.ArgumentParser(description="Run cached TileLang kernel (direct .so)")
    parser.add_argument("--cache_dir", nargs='?', default="/home/chenxi/.tilelang/cache/mha_fwd_bhsd2", help="cache dir containing kernel_lib.so and wrapped_kernel.cu")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--stats", action="store_true", default=True, help="print simple output stats")
    parser.add_argument("--warmup", type=int, default=100, help="warmup iterations")
    parser.add_argument("--repeat", type=int, default=200, help="repeat iterations for averaging")
    parser.add_argument("--tflops", action="store_true", default=True, help="print TFLOPs estimate")
    args = parser.parse_args()

    params = parse_wrapped_kernel_params(args.cache_dir)
    k = KernelLib(args.cache_dir)
    k.load()
    k.init()
    Q, K, V, O = k.allocate_random_inputs(params, device=args.device)

    # one run to materialize allocations/kernels
    k.call(Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), 0)
    torch.cuda.synchronize()

    avg_ms = _measure_latency_ms(k, Q, K, V, O, warmup=args.warmup, repeat=args.repeat)

    if args.stats:
        print(f"output: shape={tuple(O.shape)}, dtype={O.dtype}, device={O.device}")
        print(f"latency: {avg_ms:.2f} ms (warmup={args.warmup}, repeat={args.repeat})")
        if args.tflops:
            b, h, s, d = O.shape
            total_flops = 4.0 * b * h * s * s * d
            tflops = total_flops / avg_ms * 1e-9
            print(f"throughput: {tflops:.2f} TFlops")

if __name__ == "__main__":
    main()
