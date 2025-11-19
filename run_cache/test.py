import argparse
import math
import torch
import torch.nn.functional as F
try:
    from .runner import KernelLib, parse_wrapped_kernel_params, recompile_cache
except ImportError:
    # 允许脚本直接执行: 将项目根目录加入 sys.path 后使用绝对导入
    import sys, os
    # __file__ = /home/chenxi/tilelang/cache/test.py
    # 需要加入 /home/chenxi (项目根)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from tilelang.cache.runner import KernelLib, parse_wrapped_kernel_params, recompile_cache


def check_correctness_with_sdpa(Q, K, V, O_tl, params, atol=1e-2, rtol=1e-4):
    """
    使用 PyTorch 的 scaled_dot_product_attention 验证 TileLang kernel 输出的正确性。
    
    Args:
        Q, K, V: 输入张量，shape 为 (batch, heads, seq, dim)
        O_tl: TileLang kernel 的输出，shape 为 (batch, heads, seq, dim)
        params: KernelParams 对象
        atol: 绝对误差容忍度
        rtol: 相对误差容忍度
    
    Returns:
        bool: 是否通过验证
    """
    scale = 1.0 / math.sqrt(params.dim)
    
    # 使用 PyTorch SDPA 计算参考输出
    with torch.no_grad():
        torch.backends.cuda.enable_flash_sdp(enabled=True)
        O_ref = F.scaled_dot_product_attention(
            Q, K, V, 
            scale=scale,
            is_causal=False  # 默认非 causal，如果需要可以添加参数
        )
    
    try:
        torch.testing.assert_close(O_tl.cpu(), O_ref.cpu(), atol=atol, rtol=rtol)
        max_diff = (O_tl - O_ref).abs().max().item()
        mean_diff = (O_tl - O_ref).abs().mean().item()
        print(f"✓ 正确性验证通过!")
        print(f"  最大误差: {max_diff:.6f}")
        print(f"  平均误差: {mean_diff:.6f}")
        print(f"  容忍度: atol={atol}, rtol={rtol}")
        return True
    except AssertionError as e:
        max_diff = (O_tl - O_ref).abs().max().item()
        mean_diff = (O_tl - O_ref).abs().mean().item()
        print(f"✗ 正确性验证失败!")
        print(f"  最大误差: {max_diff:.6f}")
        print(f"  平均误差: {mean_diff:.6f}")
        print(f"  容忍度: atol={atol}, rtol={rtol}")
        print(f"  错误详情: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run cached TileLang kernel (direct .so)")
    parser.add_argument("--cache_dir", nargs='?', default="/home/chenxi/.tilelang/cache/mha_fwd_bhsd", help="cache dir with kernel_lib.so and wrapped_kernel.cu")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--stats", action="store_true", help="print simple output stats")
    parser.add_argument("--rebuild", action="store_true", help="force rebuild via nvcc before running")
    parser.add_argument("--arch", default=None, help="gpu arch, e.g., sm_90; default: auto-detect")
    parser.add_argument("--fast_math", action="store_true", help="enable nvcc --use_fast_math when rebuilding")
    parser.add_argument("--maxrregcount", type=int, default=None, help="cap register count via nvcc --maxrregcount")
    parser.add_argument("--check", action="store_true", help="check correctness against PyTorch SDPA")
    parser.add_argument("--atol", type=float, default=1e-2, help="absolute tolerance for correctness check")
    parser.add_argument("--rtol", type=float, default=1e-4, help="relative tolerance for correctness check")
    args = parser.parse_args()

    if args.rebuild:
        extra_flags = []
        if args.fast_math:
            extra_flags.append("--use_fast_math")
        if args.maxrregcount is not None:
            extra_flags.append(f"--maxrregcount={args.maxrregcount}")
        recompile_cache(args.cache_dir, arch=args.arch, extra_nvcc_flags=extra_flags)

    params = parse_wrapped_kernel_params(args.cache_dir)
    k = KernelLib(args.cache_dir)
    k.load()
    k.init()
    Q, K, V, O = k.allocate_random_inputs(params, device=args.device)
    O = k.run_with_tensors(Q, K, V, O)

    if args.stats:
        print(f"output: shape={tuple(O.shape)}, dtype={O.dtype}, device={O.device}")
        print(f"min={O.min().item():.4f} max={O.max().item():.4f} mean={O.mean().item():.4f} std={O.std().item():.4f}")
    
    if args.check:
        print(f"\n开始正确性验证 (对比 PyTorch SDPA)...")
        print(f"输入形状: batch={params.batch}, heads={params.heads}, seq={params.seq}, dim={params.dim}")
        check_correctness_with_sdpa(Q, K, V, O, params, atol=args.atol, rtol=args.rtol)

if __name__ == "__main__":
    main()
