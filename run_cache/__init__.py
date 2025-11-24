"""
通用缓存 kernel 调用工具包。

- runner.py: 提供加载、参数解析、执行等核心功能
- cli.py: 简单命令行入口
"""

from .runner import (
    KernelLib,
    OldKernelLib,
    NewKernelLib,
    KernelParams,
    parse_wrapped_kernel_params,
    parse_wrapped_kernel_params_old,
    parse_wrapped_kernel_params_new,
    infer_required_shape,
    detect_cache_format,
    recompile_cache,
    recompile_cache_old,
    recompile_cache_new,
    create_kernel_and_inputs,
)

__all__ = [
    "KernelParams",
    "KernelLib",
    "OldKernelLib",
    "NewKernelLib",
    "parse_wrapped_kernel_params",
    "parse_wrapped_kernel_params_old",
    "parse_wrapped_kernel_params_new",
    "infer_required_shape",
    "detect_cache_format",
    "recompile_cache",
    "recompile_cache_old",
    "recompile_cache_new",
    "create_kernel_and_inputs",
]


