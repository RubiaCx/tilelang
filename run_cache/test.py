import argparse
try:
    from .runner import KernelLib, parse_wrapped_kernel_params, recompile_cache
except ImportError:
    # 允许脚本直接执行: 将项目根目录加入 sys.path 后使用绝对导入
    import sys, os
    # __file__ = /home/chenxi/tilelang/cache/test.py
    # 需要加入 /home/chenxi (项目根)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from tilelang.cache.runner import KernelLib, parse_wrapped_kernel_params, recompile_cache


def main():
    parser = argparse.ArgumentParser(description="Run cached TileLang kernel (direct .so)")
    parser.add_argument("--cache_dir", nargs='?', default="/home/chenxi/.tilelang/cache/mha_fwd_bhsd", help="cache dir with kernel_lib.so and wrapped_kernel.cu")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--stats", action="store_true", help="print simple output stats")
    parser.add_argument("--rebuild", action="store_true", help="force rebuild via nvcc before running")
    parser.add_argument("--arch", default=None, help="gpu arch, e.g., sm_90; default: auto-detect")
    args = parser.parse_args()

    if args.rebuild:
        recompile_cache(args.cache_dir, arch=args.arch)

    params = parse_wrapped_kernel_params(args.cache_dir)
    k = KernelLib(args.cache_dir)
    k.load()
    k.init()
    Q, K, V, O = k.allocate_random_inputs(params, device=args.device)
    O = k.run_with_tensors(Q, K, V, O)

    if args.stats:
        print(f"output: shape={tuple(O.shape)}, dtype={O.dtype}, device={O.device}")
        print(f"min={O.min().item():.4f} max={O.max().item():.4f} mean={O.mean().item():.4f} std={O.std().item():.4f}")

if __name__ == "__main__":
    main()
