#!/usr/bin/env python3
# ruff: noqa
import argparse
import math
import os
import sys
from importlib.machinery import SourceFileLoader

import torch

import tilelang
from tilelang.profiler import do_bench

def load_tilelang_flash_fwd():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    src = os.path.join(
        repo_root,
        "examples",
        "flash_attention",
        "example_mha_fwd_bhsd_wgmma_pipelined.py",
    )
    src = os.path.abspath(src)
    mod = SourceFileLoader("flash_attn", src).load_module()
    return mod.flashattn

def maybe_load_flash_attn_func():
    try:
        from flash_attn_interface import flash_attn_func 
        return flash_attn_func
    except Exception:
        # Try from local flash-attention repo located next to tilelang root: <parent_of_repo_root>/flash-attention/hopper
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(here))
        parent = os.path.dirname(repo_root)
        repo_test = os.path.join(parent, "flash-attention", "hopper")
        if os.path.isdir(repo_test):
            if repo_test not in sys.path:
                sys.path.insert(0, repo_test)
            try:
                from flash_attn_interface import flash_attn_func 
                return flash_attn_func
            except Exception:
                return None
        return None


def compute_total_flops_fwd(batch: int, heads: int, seqlen: int, dim: int, causal: bool) -> float:
    # 2 matmuls: QK^T and P@V
    flops_per_matmul = 2.0 * batch * heads * seqlen * seqlen * dim
    total = 2 * flops_per_matmul
    if causal:
        total *= 0.5
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--block_m", type=int, default=128)
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    device = "cuda"
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.manual_seed(0)

    B, H, S, D = args.batch, args.heads, args.seqlen, args.dim
    total_flops = compute_total_flops_fwd(B, H, S, D, args.causal)

    # Inputs: match TileLang kernel layout [batch, heads, seq, dim]
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # TileLang kernel (forward)
    flashattn = load_tilelang_flash_fwd()
    # flashattn signature:
    #   flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, block_M=..., block_N=..., num_stages=..., threads=...)
    # Here we benchmark seq_q = seq_kv = S.
    tl_kernel = flashattn(
        B,
        H,
        S,
        S,
        D,
        args.causal,
        block_M=args.block_m,
        block_N=args.block_n,
    )

    # Run once to materialize
    with torch.no_grad():
        tl_out = tl_kernel(Q, K, V)
    torch.cuda.synchronize()

    def run_tilelang():
        tl_kernel(Q, K, V)

    tl_ms = do_bench(run_tilelang, warmup=args.warmup, rep=args.rep)
    print(f"TileLang: {tl_ms:.2f} ms")
    print(f"TileLang: {total_flops / tl_ms * 1e-9:.2f} TFlops")

    # Optional: flash-attention library baseline
    flash_attn_func = maybe_load_flash_attn_func()
    if flash_attn_func is not None:
        # flash-attn expects (B, Q, H, D) and (B, K, H, D) by many wrappers; keep consistent with their interface
        q = Q.detach().clone().requires_grad_(False)
        k = K.detach().clone().requires_grad_(False)
        v = V.detach().clone().requires_grad_(False)

        with torch.no_grad():
            out, lse = flash_attn_func(q, k, v, causal=args.causal, num_splits=0)
        torch.cuda.synchronize()

        def run_flash():
            flash_attn_func(q, k, v, causal=args.causal, num_splits=0)

        fa_ms = do_bench(run_flash, warmup=args.warmup, rep=args.rep)
        print(f"Flash-Attn: {fa_ms:.2f} ms")
        print(f"Flash-Attn: {total_flops / fa_ms * 1e-9:.2f} TFlops")
        print(f"speedup: {fa_ms / tl_ms:.2f}x (TileLang vs Flash-Attn)")
    else:
        # Torch reference (slow)
        scale = 1.0 / math.sqrt(D)

        def run_torch():
            attn = torch.softmax((Q.float() @ K.float().transpose(1, 2)) * scale, dim=-1).to(Q.dtype)
            _ = attn @ V

        ref_ms = do_bench(run_torch, warmup=max(10, args.warmup // 10), rep=max(10, args.rep // 5))
        print(f"Torch ref: {ref_ms:.2f} ms")
        print(f"Torch ref: {total_flops / ref_ms * 1e-9:.2f} TFlops")
        print(f"speedup: {ref_ms / tl_ms:.2f}x (TileLang vs Torch)")


if __name__ == "__main__":
    main()


