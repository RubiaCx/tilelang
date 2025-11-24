import os
import re
import ctypes
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch.utils import dlpack as torch_dlpack
import subprocess

try:
    from tilelang import tvm
    from tvm import runtime
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False


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
    解析 kernel 期望的形状与 dtype，自动检测格式。
    """
    format_type = detect_cache_format(cache_dir)
    if format_type == "old":
        return parse_wrapped_kernel_params_old(cache_dir)
    else:
        return parse_wrapped_kernel_params_new(cache_dir)


def parse_wrapped_kernel_params_old(cache_dir: str) -> KernelParams:
    """
    从旧格式的 wrapped_kernel.cu 解析 kernel 期望的形状与 dtype。
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


def detect_cache_format(cache_dir: str) -> Literal["old", "new"]:
    """
    检测 cache 目录的格式。
    
    Returns:
        "old": 旧格式，有 wrapped_kernel.cu 和 kernel_lib.so
        "new": 新格式，有 host_kernel.cu、device_kernel.cu 和 executable.so
    """
    has_wrapped = os.path.exists(os.path.join(cache_dir, "wrapped_kernel.cu"))
    has_kernel_lib = os.path.exists(os.path.join(cache_dir, "kernel_lib.so"))
    has_host = os.path.exists(os.path.join(cache_dir, "host_kernel.cu"))
    has_device = os.path.exists(os.path.join(cache_dir, "device_kernel.cu"))
    has_executable = os.path.exists(os.path.join(cache_dir, "executable.so"))
    
    if has_wrapped and has_kernel_lib:
        return "old"
    elif has_host and has_device and has_executable:
        return "new"
    else:
        raise RuntimeError(
            f"无法识别 cache 格式。目录: {cache_dir}\n"
            f"旧格式需要: wrapped_kernel.cu, kernel_lib.so\n"
            f"新格式需要: host_kernel.cu, device_kernel.cu, executable.so"
        )


def parse_wrapped_kernel_params_new(cache_dir: str) -> KernelParams:
    """
    从新格式的 params.pkl 解析 kernel 期望的形状与 dtype。
    params.pkl 存储的是 list[KernelParam]，每个 KernelParam 有 dtype 和 shape。
    对于 MHA kernel，通常前 3 个参数是 Q, K, V（shape 相同），第 4 个是 Output。
    """
    import pickle
    params_path = os.path.join(cache_dir, "params.pkl")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.pkl not found under: {cache_dir}")
    
    try:
        with open(params_path, "rb") as f:
            kernel_params = pickle.load(f)
        
        # kernel_params 应该是 list[KernelParam]
        # 对于 MHA，通常有 4 个参数：Q, K, V, Output
        # Q, K, V 的 shape 应该相同，格式为 (batch, heads, seq, dim)
        if isinstance(kernel_params, list) and len(kernel_params) >= 3:
            # 使用第一个参数（Q）的 shape 和 dtype
            q_param = kernel_params[0]
            if hasattr(q_param, 'shape') and hasattr(q_param, 'dtype'):
                shape = q_param.shape
                dtype = q_param.dtype
                
                # shape 应该是 [batch, heads, seq, dim] 或类似的格式
                # 需要过滤掉 Var 类型的动态维度
                shape_ints = [s for s in shape if isinstance(s, int)]
                if len(shape_ints) >= 4:
                    batch, heads, seq, dim = shape_ints[:4]
                    return KernelParams(batch=batch, heads=heads, seq=seq, dim=dim, dtype=dtype)
                elif len(shape_ints) == 4:
                    # 如果恰好是 4 个整数
                    batch, heads, seq, dim = shape_ints
                    return KernelParams(batch=batch, heads=heads, seq=seq, dim=dim, dtype=dtype)
    except Exception as e:
        # 如果 pickle 加载失败，尝试从 host_kernel.cu 解析
        pass
    
    # 如果无法从 params.pkl 读取，尝试从 host_kernel.cu 解析
    host_path = os.path.join(cache_dir, "host_kernel.cu")
    if os.path.exists(host_path):
        with open(host_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        
        # 尝试从 host_kernel.cu 中解析 shape 信息
        # 查找 shape 相关的模式
        shape_patterns = [
            r"shape\[0\]\s*=\s*(\d+)",
            r"shape\[1\]\s*=\s*(\d+)",
            r"shape\[2\]\s*=\s*(\d+)",
            r"shape\[3\]\s*=\s*(\d+)",
        ]
        shapes = []
        for pattern in shape_patterns:
            m = re.search(pattern, code)
            if m:
                shapes.append(int(m.group(1)))
        
        if len(shapes) >= 4:
            # 假设格式是 (batch, heads, seq, dim)
            batch, heads, seq, dim = shapes[:4]
            dtype = torch.float16  # 默认
            return KernelParams(batch=batch, heads=heads, seq=seq, dim=dim, dtype=dtype)
    
    raise RuntimeError(f"无法从 params.pkl 或 host_kernel.cu 解析参数: {cache_dir}")


class OldKernelLib:
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


class NewKernelLib:
    """新格式的 kernel loader，使用 TVM runtime。"""
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.executable_path = os.path.join(cache_dir, "executable.so")
        self._rt_mod = None
        self._main_func = None
    
    def load(self) -> None:
        if not TVM_AVAILABLE:
            raise RuntimeError("TVM 不可用，无法加载新格式的 kernel。请安装 tilelang/tvm。")
        if not os.path.exists(self.executable_path):
            raise FileNotFoundError(f"executable.so not found under: {self.executable_path}")
        
        # 加载 TVM runtime module
        self._rt_mod = runtime.load_module(self.executable_path)
        # 获取 main 函数
        self._main_func = self._rt_mod["main"]
        if self._main_func is None:
            raise RuntimeError("无法找到 main 函数")
    
    def init(self) -> None:
        # 新格式可能不需要显式初始化，但保留接口兼容性
        pass
    
    def run_with_tensors(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Output: Optional[torch.Tensor] = None, stream_ptr: Optional[int] = 0) -> torch.Tensor:
        if Output is None:
            Output = torch.empty_like(Q)
        
        # 将 PyTorch tensor 转换为 TVM NDArray（通过 DLPack）
        def to_tvm_ndarray(t: torch.Tensor):
            # 兼容旧版 PyTorch，没有 Tensor.to_dlpack 方法
            return runtime.from_dlpack(torch_dlpack.to_dlpack(t.contiguous()))
        
        Q_tvm = to_tvm_ndarray(Q)
        K_tvm = to_tvm_ndarray(K)
        V_tvm = to_tvm_ndarray(V)
        O_tvm = to_tvm_ndarray(Output)
        
        # 调用 main 函数
        self._main_func(Q_tvm, K_tvm, V_tvm, O_tvm)
        
        torch.cuda.synchronize()
        return Output
    
    def allocate_random_inputs(self, params: KernelParams, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = infer_required_shape(params)
        Q = torch.randn(*shape, dtype=params.dtype, device=device)
        K = torch.randn(*shape, dtype=params.dtype, device=device)
        V = torch.randn(*shape, dtype=params.dtype, device=device)
        O = torch.empty_like(Q)
        return Q, K, V, O


class KernelLib:
    """统一的 kernel loader，自动检测格式并使用对应的实现。"""
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.format = detect_cache_format(cache_dir)
        if self.format == "old":
            self._impl = OldKernelLib(cache_dir)
        else:
            self._impl = NewKernelLib(cache_dir)
    
    def load(self) -> None:
        self._impl.load()
    
    def init(self) -> None:
        self._impl.init()
    
    def run_with_tensors(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Output: Optional[torch.Tensor] = None, stream_ptr: Optional[int] = 0) -> torch.Tensor:
        return self._impl.run_with_tensors(Q, K, V, Output, stream_ptr)
    
    def allocate_random_inputs(self, params: KernelParams, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._impl.allocate_random_inputs(params, device)


def create_kernel_and_inputs(
    cache_dir: str,
    device: str = "cuda",
) -> Tuple[KernelParams, KernelLib, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    统一的辅助函数:
      - 解析 cache 中的 kernel 参数
      - 创建并初始化 KernelLib
      - 在指定 device 上分配随机 Q/K/V/O
    """
    params = parse_wrapped_kernel_params(cache_dir)
    k = KernelLib(cache_dir)
    k.load()
    k.init()
    Q, K, V, O = k.allocate_random_inputs(params, device=device)
    return params, k, Q, K, V, O


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
    重新编译 cache 目录下的 kernel，自动检测格式。
    - 旧格式: wrapped_kernel.cu -> kernel_lib.so
    - 新格式: host_kernel.cu + device_kernel.cu -> executable.so
    """
    format_type = detect_cache_format(cache_dir)
    if format_type == "old":
        recompile_cache_old(cache_dir, arch, extra_nvcc_flags)
    else:
        recompile_cache_new(cache_dir, arch, extra_nvcc_flags)


def recompile_cache_old(cache_dir: str, arch: Optional[str] = None, extra_nvcc_flags: Optional[list] = None) -> None:
    """
    使用 nvcc 重新编译旧格式 cache 目录下的 wrapped_kernel.cu -> kernel_lib.so。
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


def recompile_cache_new(cache_dir: str, arch: Optional[str] = None, extra_nvcc_flags: Optional[list] = None) -> None:
    """
    使用 nvcc 重新编译新格式 cache 目录下的 host_kernel.cu + device_kernel.cu -> executable.so。
    - 自动探测 include 路径
    - 自动探测 GPU 架构（如未指定 arch）
    """
    host_kernel = os.path.join(cache_dir, "host_kernel.cu")
    device_kernel = os.path.join(cache_dir, "device_kernel.cu")
    out_so = os.path.join(cache_dir, "executable.so")
    
    if not os.path.exists(host_kernel):
        raise FileNotFoundError(f"host_kernel.cu not found: {host_kernel}")
    if not os.path.exists(device_kernel):
        raise FileNotFoundError(f"device_kernel.cu not found: {device_kernel}")

    # 计算架构
    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"

    # 检查是否使用 warpgroup
    try:
        with open(device_kernel, "r", encoding="utf-8", errors="ignore") as _f:
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

    # CUTLASS include 路径候选
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
        alt_candidates = [
            "/home/chenxi/flash-attention/csrc/cutlass/include",
            "/home/chenxi/tritonbench/submodules/flash-attention/csrc/cutlass/include",
            "/home/chenxi/tritonbench/submodules/xformers/third_party/cutlass/include",
        ]
        cutlass_includes = [p for p in alt_candidates if os.path.exists(os.path.join(p, "cutlass", "numeric_types.h"))]

    # TVM include 路径
    tvm_include_candidates = [
        "/home/chenxi/tilelang/3rdparty/tvm/include",
        "/home/chenxi/miniconda3/envs/triton/lib/python3.12/site-packages/tvm/include",
        "/home/chenxi/miniconda3/envs/triton_meta/lib/python3.12/site-packages/tvm/include",
    ]
    tvm_includes = [p for p in tvm_include_candidates if os.path.exists(os.path.join(p, "tvm", "runtime"))]

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    cuda_include = os.path.join(cuda_home, "include")

    # 编译命令：需要同时编译 host 和 device kernel
    cmd = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        host_kernel,
        device_kernel,
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
    for inc in tvm_includes:
        cmd.append(f"-I{inc}")
    if extra_nvcc_flags:
        cmd.extend(extra_nvcc_flags)

    env = os.environ.copy()
    lib64 = os.path.join(cuda_home, "lib64")
    env["LIBRARY_PATH"] = f"{lib64}:{env.get('LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{lib64}:{env.get('LD_LIBRARY_PATH', '')}"

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed (code {proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


