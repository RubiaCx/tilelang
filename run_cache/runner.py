import os
import re
import ctypes
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import subprocess


@dataclass
class KernelParams:
    # 约定 shape 顺序为 (batch, heads, seq, dim)
    batch: int
    heads: int
    seq: int
    dim: int
    dtype: torch.dtype = torch.float16


def _parse_datatype(code: str) -> torch.dtype:
    # 粗略映射 CUtensorMapDataType -> torch.dtype
    # 仅处理已知的 6(fp16) 与 0(fp32)，其他默认 fp16
    m = re.search(r"\b(Q|K|V)_desc_type\s*=\s*\(CUtensorMapDataType\)(\d+)\s*;", code)
    if not m:
        return torch.float16
    v = int(m.group(2))
    if v == 6:
        return torch.float16
    if v == 0:
        return torch.float32
    return torch.float16


def parse_wrapped_kernel_params(cache_dir: str) -> KernelParams:
    """
    从 wrapped_kernel.cu 解析 kernel 期望的形状与 dtype。
    该文件中 globalDim 顺序通常为 {dim, seq, heads, batch}
    """
    wrapped_path = os.path.join(cache_dir, "wrapped_kernel.cu")
    if not os.path.exists(wrapped_path):
        raise FileNotFoundError(f"wrapped_kernel.cu not found under: {cache_dir}")

    with open(wrapped_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    m = re.search(r"Q_desc_globalDim\[4\]\s*=\s*\{([^}]+)\};", code)
    if not m:
        # 退化到 K/V 之一
        m = re.search(r"(K|V)_desc_globalDim\[4\]\s*=\s*\{([^}]+)\};", code)
        if not m:
            raise RuntimeError("Failed to parse *_desc_globalDim from wrapped_kernel.cu")
        dims_str = m.group(2)
    else:
        dims_str = m.group(1)

    dims = [int(x.strip()) for x in dims_str.split(',')]
    if len(dims) != 4:
        raise RuntimeError(f"Unexpected globalDim: {dims}")

    # code 中顺序为 {dim, seq, heads, batch} -> 转为 (batch, heads, seq, dim)
    dim, seq, heads, batch = dims
    dtype = _parse_datatype(code)
    return KernelParams(batch=batch, heads=heads, seq=seq, dim=dim, dtype=dtype)


def infer_required_shape(params: KernelParams) -> Tuple[int, int, int, int]:
    return params.batch, params.heads, params.seq, params.dim


class KernelLib:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.lib_path = os.path.join(cache_dir, "kernel_lib.so")
        self._lib = None
        self._init = None
        self._call = None
        self._get_last_error = None

    def load(self) -> None:
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"kernel_lib.so not found under: {self.lib_path}")
        self._lib = ctypes.CDLL(self.lib_path)
        self._init = self._lib.init
        self._init.restype = ctypes.c_int
        self._call = self._lib.call
        self._call.restype = ctypes.c_int
        self._call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self._get_last_error = self._lib.get_last_error
        self._get_last_error.restype = ctypes.c_char_p

    def get_last_error(self) -> str:
        if self._get_last_error is None:
            return ""
        msg = self._get_last_error()
        return msg.decode("utf-8") if msg else ""

    def init(self) -> None:
        ret = self._init()
        if ret != 0:
            raise RuntimeError(f"init() failed: {self.get_last_error()}")

    def call(self, Q_ptr: int, K_ptr: int, V_ptr: int, O_ptr: int, stream_ptr: Optional[int] = 0) -> None:
        ret = self._call(Q_ptr, K_ptr, V_ptr, O_ptr, ctypes.c_void_p(stream_ptr or 0))
        if ret != 0:
            raise RuntimeError(f"call() failed: {self.get_last_error()}")

    def run_with_tensors(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Output: Optional[torch.Tensor] = None, stream_ptr: Optional[int] = 0) -> torch.Tensor:
        if Output is None:
            Output = torch.empty_like(Q)
        self.call(Q.data_ptr(), K.data_ptr(), V.data_ptr(), Output.data_ptr(), stream_ptr)
        torch.cuda.synchronize()
        return Output

    def allocate_random_inputs(self, params: KernelParams, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = infer_required_shape(params)
        Q = torch.randn(*shape, dtype=params.dtype, device=device)
        K = torch.randn(*shape, dtype=params.dtype, device=device)
        V = torch.randn(*shape, dtype=params.dtype, device=device)
        O = torch.empty_like(Q)
        return Q, K, V, O


def _find_tl_include_candidates() -> list:
    # 优先使用本地源码，再回退到 site-packages 等候选
    candidates = [
        "/home/chenxi/tilelang/src",
        "/home/chenxi/miniconda3/envs/triton/lib/python3.12/site-packages/tilelang/src",
        "/home/chenxi/miniconda3/envs/triton_meta/lib/python3.12/site-packages/tilelang/src",
        "/home/chenxi/miniconda3/envs/TA/lib/python3.10/site-packages/tilelang/src",
        "/home/chenxi/AttentionEngine/3rd_parties/tilelang/src",
    ]
    existing = [p for p in candidates if os.path.exists(os.path.join(p, "tl_templates", "cuda", "gemm.h"))]
    return existing


def recompile_cache(cache_dir: str, arch: Optional[str] = None, extra_nvcc_flags: Optional[list] = None) -> None:
    """
    使用 nvcc 重新编译 cache 目录下的 wrapped_kernel.cu -> kernel_lib.so。
    - 自动探测 include 路径
    - 自动探测 GPU 架构（如未指定 arch）
    """
    wrapped = os.path.join(cache_dir, "wrapped_kernel.cu")
    out_so = os.path.join(cache_dir, "kernel_lib.so")
    if not os.path.exists(wrapped):
        raise FileNotFoundError(f"wrapped_kernel.cu not found: {wrapped}")

    # 计算架构
    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"

    # 如果内核使用了 warpgroup/setmaxnreg，则需要 90a 架构
    try:
        with open(wrapped, "r", encoding="utf-8", errors="ignore") as _f:
            _code = _f.read()
        uses_warpgroup = ("warpgroup_reg_alloc" in _code) or ("warpgroup_reg_dealloc" in _code) or ("setmaxnreg" in _code)
        if uses_warpgroup and arch.strip().lower() == "sm_90":
            arch = "sm_90a"
    except Exception:
        pass

    # include 路径
    tl_includes = _find_tl_include_candidates()
    if not tl_includes:
        raise RuntimeError("Cannot locate tl_templates include root (tilelang/src)")

    # CUTLASS include 路径候选（优先使用本地/随包提供版本）
    cutlass_include_candidates = [
        "/home/chenxi/tilelang/3rdparty/cutlass/include",
        "/home/chenxi/tilelang/3rdparty/tvm/3rdparty/cutlass/include",
        "/home/chenxi/miniconda3/envs/triton/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include",
        "/home/chenxi/miniconda3/envs/triton_meta/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include",
        "/home/chenxi/miniconda3/envs/TA/lib/python3.10/site-packages/tilelang/3rdparty/cutlass/include",
        "/home/chenxi/tvm-tl/3rdparty/cutlass/include",
        "/home/chenxi/tvm-tl/python/tvm/3rdparty/cutlass/include",
        "/home/chenxi/cutlass/include",
    ]
    cutlass_includes = [p for p in cutlass_include_candidates if os.path.exists(os.path.join(p, "cutlass", "numeric_types.h"))]
    if not cutlass_includes:
        # 容错：有些安装路径在第三方工程内
        alt_candidates = [
            "/home/chenxi/flash-attention/csrc/cutlass/include",
            "/home/chenxi/tritonbench/submodules/flash-attention/csrc/cutlass/include",
            "/home/chenxi/tritonbench/submodules/xformers/third_party/cutlass/include",
        ]
        cutlass_includes = [p for p in alt_candidates if os.path.exists(os.path.join(p, "cutlass", "numeric_types.h"))]

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    cuda_include = os.path.join(cuda_home, "include")

    cmd = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        wrapped,
        "-o",
        out_so,
        f"-gencode=arch=compute_{arch.split('_')[1]},code={arch}",
        f"-I{cuda_include}",
        "-lcudart",
        "-lcuda",
    ]
    for inc in tl_includes:
        cmd.append(f"-I{inc}")
    for inc in cutlass_includes:
        cmd.append(f"-I{inc}")
    if extra_nvcc_flags:
        cmd.extend(extra_nvcc_flags)

    env = os.environ.copy()
    # 确保能找到 libcuda/libcudart
    lib64 = os.path.join(cuda_home, "lib64")
    env["LIBRARY_PATH"] = f"{lib64}:{env.get('LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{lib64}:{env.get('LD_LIBRARY_PATH', '')}"

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed (code {proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


