#!/usr/bin/env python3
"""
批量测试不同序列长度的Flash Attention性能
只输出时间结果，不输出TFlops
"""

import torch
import sys
import os
import subprocess
import time

def test_seq_length(seq_len, script_name):
    """测试指定序列长度的性能"""
    try:
        # 运行脚本并捕获输出
        cmd = f"python {script_name} --seq_len {seq_len}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return None, f"Error: {result.stderr}"
        
        # 解析输出获取Tile-lang时间
        lines = result.stdout.split('\n')
        tilelang_time = None
        
        for line in lines:
            if 'Tile-lang:' in line and 'ms' in line:
                # 提取时间值
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'ms' in part:
                        time_str = parts[i-1]
                        try:
                            tilelang_time = float(time_str)
                        except:
                            continue
                        break
        
        return tilelang_time, None
        
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)

def main():
    print("Flash Attention 序列长度性能测试")
    print("=" * 50)
    
    # 测试的序列长度
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    # 要测试的脚本
    scripts = [
        ("基础版本", "example_mha_fwd_bshd.py"),
        ("WGMMA流水线版本", "example_mha_fwd_bshd_wgmma_pipelined.py")
    ]
    
    results = {}
    
    for script_name, script_file in scripts:
        print(f"\n测试 {script_name} ({script_file}):")
        print("-" * 40)
        results[script_name] = {}
        
        for seq_len in seq_lengths:
            print(f"seq_len={seq_len:>4} ...", end=" ", flush=True)
            
            time_ms, error = test_seq_length(seq_len, script_file)
            
            if time_ms is not None:
                print(f"{time_ms:>8.6f} ms")
                results[script_name][seq_len] = time_ms
            else:
                print(f"失败: {error}")
                results[script_name][seq_len] = None
    
    # 输出汇总表格
    print("\n" + "=" * 70)
    print("汇总结果 (时间单位: ms)")
    print("=" * 70)
    
    # 表头
    print(f"{'seq_len':>8}", end="")
    for script_name, _ in scripts:
        print(f"{script_name:>20}", end="")
    print()
    
    print("-" * 70)
    
    # 数据行
    for seq_len in seq_lengths:
        print(f"{seq_len:>8}", end="")
        for script_name, _ in scripts:
            time_ms = results[script_name].get(seq_len)
            if time_ms is not None:
                print(f"{time_ms:>20.6f}", end="")
            else:
                print(f"{'失败':>20}", end="")
        print()
    
    print("=" * 70)

if __name__ == "__main__":
    main()
