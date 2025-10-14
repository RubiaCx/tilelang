"""
通用缓存 kernel 调用工具包。

- runner.py: 提供加载、参数解析、执行等核心功能
- cli.py: 简单命令行入口
"""

from .runner import (
    KernelLib,
    parse_wrapped_kernel_params,
    infer_required_shape,
)

__all__ = [
    "KernelLib",
    "parse_wrapped_kernel_params",
    "infer_required_shape",
]


