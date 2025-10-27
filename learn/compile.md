# 编译三阶段

| 阶段 | 名称                         | 核心文件                         | 主要工作                       |
| -- | -------------------------- | ---------------------------- | -------------------------- |
| 1  | JIT入口                      | `jit/__init__.py`            | 缓存键计算、编译触发                 |
| 2-1  | Phase-1: LowerAndLegalize  | `engine/phase.py:64`         | DSL→TIR：布局推断、TileOp降级、插入谓词 |
| 2-2 | Phase-2: OptimizeForTarget | `engine/phase.py:120`        | Hopper/Ampere 路径优化、异步拷贝注入  |
| 3  | Codegen                    | `src/target/codegen_cuda.cc` | PTX/HIP 生成与 nvrtc 编译       |


## jit

### DSL

- 路径
    - repo: [tilelang/language](https://github.com/RubiaCx/tilelang/tree/main/tilelang/language)
    - docs: [tilelang.language](https://tilelang.tile-ai.cn/autoapi/tilelang/language/index.html)

- `tilelang.language`定义写 kernel 时用到的所有 API，具体分为：
    - TileOp：TileLang 的高层算子抽象（如 `T.copy`、`T.gemm`），会在编译时降级为底层 IR
    - Fragment：寄存器中的张量片段，用于 Tensor Core 等操作

``` python
import tilelang.language as T

# 控制流与并行
with T.Kernel(T.ceildiv(M, 128), threads=256) as bx:
    with T.Parallel(128) as ty:
        with T.serial(16) as tx:
            ...

# 高层算子（TileOp）
T.copy(src, dst)           # 数据搬移
T.gemm(A, B, C)            # 矩阵乘法
T.reduce_absmax(tensor)    # 归约操作

# 内存分配
buf = T.alloc_shared([128, 64], "float16")
acc = T.alloc_fragment([8, 8], "float32")

# 原子操作
T.atomic_add(ptr, val)
```


### JIT, cache & Adapter

- 路径
    - repo: [tilelang/jit](https://github.com/RubiaCx/tilelang/tree/main/tilelang/language)
    - docs: [tilelang.jit](https://tilelang.tile-ai.cn/autoapi/tilelang/jit/index.html)

- 结构
    ``` bash
    tilelang/jit/
      ├── __init__.py         # @tilelang.jit 装饰器入口
      ├── kernel.py           # JITKernel 类 编译产物容器
      └── adapter/            # 多后端适配（cython/nvrtc/torch/dlpack）
    ```
    
#### JIT 入口

- [`tilelang.jit`](https://github.com/RubiaCx/tilelang/blob/5475f8e7fa392f1ffb854098c313e917be636246/tilelang/jit/__init__.py#L30) 为 TileLang 提供 auto-tuning 基础设施，使用 `tilelang.compile()` 或 `@tilelang.jit` 装饰器时，会触发编译流程，使用 TVM 将 TileLang TIR 编译为可运行的 JITKernel adapter，会在返回方法中进行缓存检查

#### 缓存检查

- [`KernelCache`](https://github.com/RubiaCx/tilelang/blob/5475f8e7fa392f1ffb854098c313e917be636246/tilelang/cache/kernel_cache.py#L30) 
    1. 计算 $cache key = hash(IRModule + Target + PassConfigs)$
    2. 检查 `~/.tilelang/cache` 中的 cache
     value
     a. 命中，说明找到了对应的 JITKernel，直接返回
     b. 不命中，进行编译生成新的 JITKernel

#### 编译入口

- 在 cache miss 时触发 [`JITKernel`](https://github.com/RubiaCx/tilelang/blob/5475f8e7fa392f1ffb854098c313e917be636246/tilelang/cache/kernel_cache.py#L185) 的编译，在 [`JITKernel` 的 `__init__`](https://github.com/RubiaCx/tilelang/blob/5475f8e7fa392f1ffb854098c313e917be636246/tilelang/jit/kernel.py#L121) 中 调用 `_compile_and_create_adapter` 构建 adapter，并进行编译参数准备，然后调用 [`tilelang.lower`](https://github.com/RubiaCx/tilelang/blob/5475f8e7fa392f1ffb854098c313e917be636246/tilelang/jit/kernel.py#L219) 进行编译
    - 编译参数
        - `PassContext`
            - `opt_level=3`: 最高优化级别
            - `config=pass_configs`: 用户指定的编译配置
            - `self.target`: 目标硬件上下文
        - 参数
            - `tilelang_func`: 已解析的TIR PrimFunc
            - `target`: 目标硬件 ("cuda", "hip", "cpu")
            - `enable_host_codegen`: DLPack后端需要主机代码
            - `enable_device_compile`: DLPack后端需要完整编译
    - JITKernel 的职责
        - 封装编译产物（PTX/CUBIN/so）
        - 管理缓存，避免重复编译
        - 选择适配器并执行配置
        - L2 持久化窗口（APW）

- 实际的编译流程由 `tilelang.lower()` 完成，对于编译得到的 **artifact**，通过对 execution_backend 的判断获得真正的 **adapter**，
    - execution_backend
        - cython：预编译扩展，性能最优
        - nvrtc：运行时编译，适合快速迭代
        - torch（Metal）：MPS 后端，用于 Apple Silicon
        - dlpack：零拷贝张量传递
    - 适配器模式
        - `self.adapter = adapter`把具体后端的 adapter 对象挂到实例上，便于后续查询能力、做资源管理等
        - `self.torch_function = adapter.func` 直接把 adapter 里真正“可调用”的函数句柄（callable）取出来，挂到当前实例，以后实例要执行时，就调用这个函数即可
        - 实例即可调用（callable by delegation）

## Compile Engine

## Codegen