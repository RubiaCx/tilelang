#!/usr/bin/env python3
"""
一键流水线:
  1) 重编译 cache (可关)
  2) 精度测试: 对比 PyTorch SDPA
  3) 基本性能 benchmark
  4) 可选 NCU profile
"""

import argparse
import os
import sys
from typing import Optional

import torch

try:
    # 包内运行: python -m run_cache.cli
    from .runner import (
        KernelParams,
        detect_cache_format,
        recompile_cache,
        create_kernel_and_inputs,
    )
    from .test import check_correctness_with_sdpa
    from .bench import _measure_latency_ms
except ImportError:
    # 直接脚本运行: python cli.py
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../tilelang/run_cache
    PROJECT_ROOT = os.path.dirname(THIS_DIR)                       # .../tilelang
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from run_cache.runner import (
        KernelParams,
        detect_cache_format,
        recompile_cache,
        create_kernel_and_inputs,
    )
    from run_cache.test import check_correctness_with_sdpa
    from run_cache.bench import _measure_latency_ms


def _run_rebuild(cache_dir: str, arch: Optional[str], fast_math: bool, maxrregcount: Optional[int]) -> None:
    print("\n[Step 1] Rebuild kernel")
    extra_flags: list[str] = []
    if fast_math:
        extra_flags.append("--use_fast_math")
    if maxrregcount is not None:
        extra_flags.append(f"--maxrregcount={maxrregcount}")
    recompile_cache(cache_dir, arch=arch, extra_nvcc_flags=extra_flags)
    print("  ✓ Rebuild 完成")


def _run_correctness(cache_dir: str, device: str, atol: float, rtol: float) -> None:
    print("\n[Step 2] 精度测试 (对比 PyTorch SDPA)")
    params, k, Q, K, V, O = create_kernel_and_inputs(cache_dir, device=device)
    O = k.run_with_tensors(Q, K, V, O)

    print(f"  输入形状: batch={params.batch}, heads={params.heads}, seq={params.seq}, dim={params.dim}")
    ok = check_correctness_with_sdpa(Q, K, V, O, params, atol=atol, rtol=rtol)
    if not ok:
        print("  ✗ 精度验证未通过（见上方详细信息）")
    else:
        print("  ✓ 精度验证通过")


def _run_bench(cache_dir: str, device: str, warmup: int, repeat: int) -> None:
    print("\n[Step 3] Benchmark")
    params, k, Q, K, V, O = create_kernel_and_inputs(cache_dir, device=device)

    # 预热一次，materialize 内部资源
    O = k.run_with_tensors(Q, K, V, O)
    torch.cuda.synchronize()

    avg_ms = _measure_latency_ms(k, Q, K, V, O, warmup=warmup, repeat=repeat)

    print(f"  output: shape={tuple(O.shape)}, dtype={O.dtype}, device={O.device}")
    print(f"  latency: {avg_ms:.2f} ms (warmup={warmup}, repeat={repeat})")

    # 粗略 TFLOPs 估计，与 bench.py 一致: 4 * B * H * S * S * D
    b, h, s, d = O.shape
    total_flops = 4.0 * b * h * s * s * d
    tflops = total_flops / avg_ms * 1e-9
    print(f"  throughput: {tflops:.2f} TFlops")


def _run_ncu(cache_dir: str, arch: Optional[str], out_dir: str) -> None:
    """调用 ncu 对 main_kernel 做 profile。"""
    print("\n[Step 4] NCU Profile")
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.basename(os.path.abspath(cache_dir.rstrip("/")))
    rep_path = os.path.join(out_dir, f"{base_name}.ncu-rep")
    details_txt = os.path.join(out_dir, f"{base_name}_details.txt")
    source_txt = os.path.join(out_dir, f"{base_name}_source.txt")
    summary_txt = os.path.join(out_dir, f"{base_name}_summary.txt")

    python_bin = sys.executable
    kernel_filter = "main_kernel"

    print(f"  输出目录: {out_dir}")
    print(f"  REP: {rep_path}")

    # 调用 ncu。这里直接 profile `python -m run_cache.test`，测试脚本内部会使用 cache_dir。
    cmd = [
        "ncu",
        "--set", "full",
        "--kernel-name", kernel_filter,
        "--force-overwrite",
        "--import-source", "yes",
        "-o", rep_path,
        python_bin,
        "-m",
        "run_cache.test",
        "--cache_dir", cache_dir,
        "--stats",
    ]
    if arch is not None:
        cmd.extend(["--arch", arch])

    print("  运行命令:")
    print("   ", " ".join(cmd))

    import subprocess

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("  ✗ 未找到 ncu，请确认 Nsight Compute 已安装且在 PATH 中")
        return
    except subprocess.CalledProcessError as e:
        print(f"  ✗ ncu 运行失败，返回码 {e.returncode}")
        return

    # 生成文本报告（与 ncu_profile.sh 类似）
    try:
        subprocess.run(
            ["ncu", "-i", rep_path, "--page", "details", "--print-summary", "per-kernel"],
            check=False,
            stdout=open(details_txt, "w"),
        )
        subprocess.run(
            ["ncu", "-i", rep_path, "--page", "source", "--csv"],
            check=False,
            stdout=open(source_txt, "w"),
        )
        with open(summary_txt, "w") as f:
            proc = subprocess.run(
                ["ncu", "-i", rep_path, "--print-summary", "per-kernel"],
                check=False,
                stdout=subprocess.PIPE,
                text=True,
            )
            f.write("\n".join(proc.stdout.splitlines()[:50]))
    except FileNotFoundError:
        # 如果后处理时找不到 ncu，不视为致命错误
        print("  (warning) 生成文本报告时未找到 ncu，可忽略")

    print("  ✓ NCU profile 完成")
    print(f"    REP     : {rep_path}")
    print(f"    DETAILS : {details_txt}")
    print(f"    SOURCE  : {source_txt}")
    print(f"    SUMMARY : {summary_txt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="一键 rebuild + 精度测试 + bench + (可选) NCU")
    parser.add_argument(
        "--cache_dir",
        required=True,
        help="cache 目录 (支持旧格式 wrapped_kernel.cu/kernel_lib.so 和新格式 host/device/executable)",
    )
    parser.add_argument("--device", default="cuda", help="torch device (默认: cuda)")

    # rebuild 相关
    parser.add_argument("--no_rebuild", action="store_true", help="跳过重编译 (默认会重编译)")
    parser.add_argument("--arch", default=None, help="GPU 架构, 如 sm_90; 默认自动探测")
    parser.add_argument("--fast_math", action="store_true", help="rebuild 时启用 --use_fast_math")
    parser.add_argument("--maxrregcount", type=int, default=None, help="rebuild 时传给 --maxrregcount")

    # 精度测试
    parser.add_argument("--no_check", action="store_true", help="跳过精度测试 (默认执行)")
    parser.add_argument("--atol", type=float, default=1e-2, help="精度测试 atol")
    parser.add_argument("--rtol", type=float, default=1e-4, help="精度测试 rtol")

    # bench
    parser.add_argument("--no_bench", action="store_true", help="跳过 bench (默认执行)")
    parser.add_argument("--warmup", type=int, default=100, help="bench warmup 次数")
    parser.add_argument("--repeat", type=int, default=200, help="bench repeat 次数")

    # NCU
    parser.add_argument("--ncu", action="store_true", help="最后执行 NCU profile")
    parser.add_argument(
        "--ncu_out_dir",
        default="/home/chenxi/tilelang/cache/ncu_results",
        help="NCU 输出目录 (默认: /home/chenxi/tilelang/cache/ncu_results)",
    )

    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    print(f"cache_dir = {cache_dir}")

    # 检测格式
    cache_format = detect_cache_format(cache_dir)
    format_name = "旧格式 (old)" if cache_format == "old" else "新格式 (new)"
    print(f"检测到 cache 格式: {format_name}")

    # Step 1: rebuild
    if not args.no_rebuild:
        _run_rebuild(cache_dir, arch=args.arch, fast_math=args.fast_math, maxrregcount=args.maxrregcount)
    else:
        print("\n[Step 1] 跳过 Rebuild")

    # Step 2: correctness
    if not args.no_check:
        _run_correctness(cache_dir, device=args.device, atol=args.atol, rtol=args.rtol)
    else:
        print("\n[Step 2] 跳过精度测试")

    # Step 3: bench
    if not args.no_bench:
        _run_bench(cache_dir, device=args.device, warmup=args.warmup, repeat=args.repeat)
    else:
        print("\n[Step 3] 跳过 bench")

    # Step 4: NCU
    if args.ncu:
        _run_ncu(cache_dir, arch=args.arch, out_dir=args.ncu_out_dir)
    else:
        print("\n[Step 4] 未启用 NCU (如需启用加上 --ncu)")


if __name__ == "__main__":
    main()


