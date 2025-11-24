#!/usr/bin/env python3
import argparse
import os
import sys

import torch

try:
    # 作为包运行: 例如 `python -m run_cache.bench`
    from .runner import (
        KernelLib,
        detect_cache_format,
        recompile_cache,
        create_kernel_and_inputs,
    )
except ImportError:
    # 允许脚本直接执行: `python bench.py`
    # 将项目根目录 (/home/chenxi/tilelang) 加到 sys.path，然后用顶层模块导入
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../tilelang/run_cache
    PROJECT_ROOT = os.path.dirname(THIS_DIR)                       # .../tilelang
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from run_cache.runner import (
        KernelLib,
        detect_cache_format,
        recompile_cache,
        create_kernel_and_inputs,
    )


def _measure_latency_ms(
    k: KernelLib,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    warmup: int,
    repeat: int,
) -> float:
    for _ in range(max(0, warmup)):
        k.run_with_tensors(Q, K, V, O)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(max(1, repeat)):
        k.run_with_tensors(Q, K, V, O)
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
    parser.add_argument("--rebuild", action="store_true", help="force rebuild via nvcc before running")
    parser.add_argument("--arch", default=None, help="gpu arch, e.g., sm_90; default: auto-detect")
    parser.add_argument("--fast_math", action="store_true", help="enable nvcc --use_fast_math when rebuilding")
    parser.add_argument("--maxrregcount", type=int, default=None, help="cap register count via nvcc --maxrregcount")
    args = parser.parse_args()

    # 检测 cache 格式
    cache_format = detect_cache_format(args.cache_dir)
    format_name = "旧格式 (old)" if cache_format == "old" else "新格式 (new)"
    print(f"检测到 cache 格式: {format_name}")

    if args.rebuild:
        extra_flags = []
        if args.fast_math:
            extra_flags.append("--use_fast_math")
        if args.maxrregcount is not None:
            extra_flags.append(f"--maxrregcount={args.maxrregcount}")
        recompile_cache(args.cache_dir, arch=args.arch, extra_nvcc_flags=extra_flags)

    params, k, Q, K, V, O = create_kernel_and_inputs(args.cache_dir, device=args.device)

    # one run to materialize allocations/kernels
    O = k.run_with_tensors(Q, K, V, O)
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
