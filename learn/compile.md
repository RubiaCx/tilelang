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

## Compile

- 在 `tilelang.lower()` 中定义了两阶段编译优化
    ``` python
    # Phase 1: Lower and legalize the IR
    mod = LowerAndLegalize(mod, target)

    # Phase 2: Optimize the IR for the target
    mod = OptimizeForTarget(mod, target)
    ```

- 结构
    ``` bash
    tilelang/engine/
      ├── phase.py      # 两阶段 Pass 流水线定义
      └── lower.py      # Host/Device 拆分与 Codegen 调度
    ```

- 具体实现使用 C++，在 [src/transform](https://github.com/RubiaCx/tilelang/tree/main/src/transform) 中

### Phase-1: LowerAndLegalize

- [`LowerAndLegalize`](https://github.com/RubiaCx/tilelang/blob/5475f8e7fa392f1ffb854098c313e917be636246/tilelang/engine/phase.py#L70) 阶段把 TIR 变成编译器能理解和优化的规范 TIR
    - ps 不同版本之间的 tilelang 还在更新前端

    ``` python
    def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
        mod = tir.transform.BindTarget(target)(mod)

    +   mod = tilelang.transform.LetInline()(mod) # Force-let inline whenever the pass config requests it.
    -   mod = tilelang.transform.FrontendLegalize()(mod) 

    +   mod = tilelang.transform.AddWrapperForSingleBufStore()(mod) # Add wrapper for single buf store
    +   mod = tilelang.transform.InjectAssumes()(mod) # Inject assumes to speedup tvm prover

        mod = tir.transform.Simplify()(mod) # Simplify the IR expressions
    
    +   mod = tilelang.transform.LayoutReducer()(mod) # Set layouts for reducers
        mod = tilelang.transform.LayoutInference()(mod) # Infer memory layouts for fragments and shared memory

        mod = tilelang.transform.LowerTileOp()(mod) # Lower high-level tile operations to low-level operations
        mod = tilelang.transform.LowerL2Persistent()(mod) # Lower l2 persistent map
        mod = tilelang.transform.LegalizeVectorizedLoop()(mod)# Legalize vectorized loops to ensure they are valid
        mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod) # Add safety checks for memory accesses
        mod = tilelang.transform.Simplify()(mod) # Simplify again to clean up any duplicated conditions, that may have been introduced by safety checks, use an enhanced pass to simplify the dynamic symbolics is merged into tvm.
        mod = tilelang.transform.LoopVectorizeDynamic()(mod)  # Try to vectorize loop with dynamic shape
    return mod
    ```

- 一共 14 个 Passes，把语义抽象的 DSL 转换为编译器可识别、可优化的标准化 TIR

| 维度  | 输入 | 输出| 说明 |
| --------- | ------------- | ------------------------| ------------------------------------------------ |
| **硬件信息** | 未绑定 | 已绑定 `Target`| 通过 `BindTarget` 注入架构与容量等上下文（如是否有 TMA、SM/共享内存大小），为 Phase-2 选择 Hopper/Ampere 路径与优化提供依据       |
| **抽象层次** | 高层算子：`T.copy`, `T.gemm` | 低层循环 + `BufferLoad` / `BufferStore`         | 完成 TileOp → IR 的语义 Lowering，显式暴露循环与访存点，便于后续匹配与改写                                           |
| **Layout** | 未确定（逻辑访问）| 已推断（`row-major` / `swizzled` / `col-major`） | 由 `LayoutInference` 固化布局与线程映射；据此可做 bank 冲突分析、选择 swizzle 策略、规划 TMA tile 形状  |
| **索引形式** | 符号式（`A[i, j]`）| 线性化（`A[linear_idx]`） | 形成可直接寻址的线性索引（常含位运算/swizzle 线性化），为向量化与存储重写做准备|
| **访存安全**   | 无越界检查  | 已插入 OOB 检查 / 谓词化 | 统一成 `if-then-else(pred, load, 0)` / 条件化 store 模式；Phase-2 可据此识别并改写为 `cp.async` / `TMA` 异步拷贝 |
| **向量化标注**  | 无或静态 | 动态条件化标注  | `LoopVectorizeDynamic` 标记候选并处理尾部拆分与对齐假设，便于 Phase-2 改写为 128-bit 访存（如 `ld/st.128`）|

#### Pass 1：BindTarget

- `BindTarget` 是 TVM 框架提供的代码，TileLang 调用这个 pass 将 Target 对象（包含硬件能力信息）绑定到 IRModule 和 PrimFunc 上，使后续的 pass 能够根据目标设备特性进行优化决策

- 使用场景
    1. `LowerAndLegalize` 阶段: 在编译流程的第一阶段，`BindTarget` 将目标设备信息附加到模块上
    2. Host 代码生成: 在生成主机端代码时,使用 `target_host` 参数调用 `BindTarget`

#### Pass 2：LetInline

- `LetInline` 在 配置项 `TL_FORCE_LET_INLINE` 为 `True` 时，会在编译流程的最开始强制执行 let 语句内联，从而减少嵌套深度，降低表达式层次，简化后续的 Simplify 和模式匹配
    ``` python
    # 内联前
    Let x = a + b in (x * x + x)
    # 内联后
    (a + b) * (a + b) + (a + b)
    ```

- 一个 LetStmt 会被内联，当且仅当满足以下条件
    - 值满足 `CanInlineLetStmt` 检查：常量、变量、是整数类型且没有副作用(side effect ≤ kPure)
    - 变量未在 buffer 定义中使用，即如果变量被用于 buffer 的 shape、strides 或其他定义字段，则不会被内联，从而避免 buffer 定义失效
    - 变量是多维 buffer 的别名，类似于 `X_shared = buffer[0:128, 0:32]`，且跨越超过 1 个非单位维度或包含向量通道，会被强制内联，避免了后续 pass(如 `Layout rewrite`、`FlattenBuffer`) 处理这些别名时出现问题
    
- 实现在 [src/transform/frontend_legalize.cc](https://github.com/RubiaCx/tilelang/blob/main/src/transform/frontend_legalize.cc)

#### Pass 3：AddWrapperForSingleBufStore

- `AddWrapperForSingleBufStore`用于为访问 fragment buffer 的单个 buffer store 语句（孤立的 `fragment[0]`）添加 `T.Parallel` 循环包装器，确保对这些 buffer 的访问符合 TileLang 的并行执行模型，用于后续的 `InjectAssumes` 和 `Simplify` Passes
   - fragment buffer 是 TileLang 中用于寄存器级别数据的特殊 buffer 类型

- 实现在 [tilelang/transform/add_bufstore_wrapper.py](https://github.com/tile-ai/tilelang/blob/main/tilelang/transform/add_bufstore_wrapper.py)
  1.  使用 `ir_transform 遍历函数体`，通过 pre-visit 和 post-visit 两个阶段处理
    a. pre-visit：跟踪 threadBinding 变量和 TileOperator 深度
    b. post-visit：对 BufferStore 节点进行转换判断，满足条件的 buffer store 语句
      - 访问 fragment buffer 且 fragment buffer 的所有索引都是 0 
      - 不在现有的 TileOperator 内部：tile_operation_depth == 0
      - 不在 threadBinding 内部
  2. 满足条件的 BufferStore 会被包装在一个 For 循环中：
    ``` python
    For(Var("_", "int32"), 0, 1, ForKind.PARALLEL, statement)
    ```
    ``` python
    # 用户写的代码
    acc = T.alloc_fragment([8, 8], "float32")
    acc[0] = initial_value  # 孤立写入

    # Pass 改写后
    for _ in T.parallel(1):  # 包装器
        acc[0] = initial_value
    ```
  
#### Pass 4：InjectAssumes

- `InjectAssumes` 为 buffer 的 shape 维度添加约束条件，，确保所有 shape 值大于 0，约束以 `AttrStmt` 节点的形式插入到 IR 中，属性名为 `tir::attr::tilelang_assume`，便于后续的 `Simplify` pass 进行更激进的优化

- TVM 的证明器在处理符号化 shape 时需要额外的约束信息，通过显式注入 "shape > 0" 的假设，可以：
  - 加速符号化表达式的简化
  - 帮助证明器验证内存访问的合法性
  - 支持更激进的优化决策，例如：
    - 消除永远为真的边界检查
    - 使用对齐的向量化加载（ldg.128 而非 ldg.64）
      ``` python
      // 注入后的伪代码
      __assume(threadIdx.x < 256);
      __assume((shared_ptr & 15) == 0);  // 16字节对齐
      ```
- 实现在 [src/transform/inject_assumes.cc](https://github.com/tile-ai/tilelang/blob/main/src/transform/inject_assumes.cc)

#### Pass 5 & 12：Simplify

- `Simplify` 使用 TVM 的 `arith::Analyzer` 作为底层简化引擎，并通过**配置选项**支持不同级别的激进优化，从而简化表达式，清理未使用的函数参数和 buffer
  - 第一次调用在 `InjectAssumes` 之后，利用注入的约束进行初步简化
  - 第二次调用在 `LegalizeSafeMemoryAccess` 之后，清理安全检查引入的冗余条件
  - **配置选项**
    - `transitively_prove_inequalities`：传递性不等式证明
    - `propagate_knowns_to_prove_conditional`：传播已知值来证明条件-
    - `propagate_knowns_to_simplify_expressions`：传播已知值来简化表达式
    - `convert_boolean_to_and_of_ors`：将布尔表达式转换为 AND-OR 形
    - `apply_constraints_to_boolean_branches`：对布尔分支应用约束

- 实现在 [src/transform/simplify.cc](https://github.com/tile-ai/tilelang/blob/main/src/transform/simplify.cc)

#### Pass 6：LayoutReducer & Pass 7：LayoutInference

- 共同负责确定和配置 TileLang 中 fragment 和 shared memory 的内存布局

- `LayoutReducer` 根据访问模式和注解，为 `local.reducer` 类(如 `T.reduce`)配置布局属性（如何在 `threadIdx.x` 上复制/分片），随后 pipeline 会将其当作 `local.fragment` 使用/进一步 Lower，这些信息随后被 `LayoutInference` 使用来推断相关 buffer 的布局
    - Reducer 的布局需要考虑线程间的数据复制和归约模式，因此需要特殊布局处理

- `LayoutInference` 收集算子与循环的 use-def 关系，基于线程绑定与目标硬件，对 fragment / shared 的布局与并行循环的线程映射/谓词/向量化进行全局推断与改写，并把结果写回到 IR 注解与循环结构中，为后续 LowerTileOp、TMA/cp.async、向量化等优化做铺垫
    1. `BufferUseDefCollector::Collect` 遍历 IR 收集所有`TileOperator`，构建 buffer 使用关系图
        1. 收集
            - 通过 `ParseOperator` 解析 Call 节点，提取 `TileOperator` 对象
            - 收集每个操作访问的 buffer，构建 buffer 使用关系图（`use_list_` 映射：buffer → 使用 buffer 的 OP 索引列表）
            - 记录每个操作的线程绑定信息 `thread_var_vec_` 和 buffer 越界状态`buffer_oob_vec_`
            - 在 Block 节点中，收集用户通过 `attr::kLayoutMap` 注解指定的布局，作为推断的起点
        2. 三级推断
          1. 首先对所有操作执行严格推断 `InferLevel::kStrict`，每个 TileOperator 的 InferLayout 方法在此级别返回强制性的布局约束 `strict_layout_map`，作为后续推断的**不可变约束**
            - GEMM 操作：根据目标架构（Volta/Ampere/Hopper）推断 A、B、C 矩阵的 fragment 布局
            - Reduce 操作：根据源 fragment 布局推断目标布局 
          2. 接下来进行通用推断 `InferLevel::kCommon`，使用 BFS 队列传播布局信息
            - 当一个 buffer 的布局被确定后，将所有使用该 buffer 的操作加入队列
            - 对队列中的操作调用 RunInferStep，尝试推断其他 buffer 的布局
            - 如果新布局与已有布局冲突，进行兼容性检查（对于 fragment buffer，检查是否为包含关系） 
          3. 对于仍未确定布局的 buffer，执行自由推断 `InferLevel::kFree`
            - 使用 UnionFind 将操作分组为连通分量，即共享 buffer 的操作在同一分量
            - 对每个分量，尝试以不同操作为根节点进行推断
            - 选择**寄存器使用量最少**的推断方案
    2. `LayoutInferencer` 类将推断的布局应用到 IR 上
      - 将 `layout_map` 附加到 Block 节点的 `attr::kLayoutMap` 注解中
      - 对于 `for_map` 中的循环（通常是 ParallelOp 生成的），执行：
        -  根据 fragment 布局，通过 PartitionLoop循环分区到线程
        - 如果循环访问非本地 buffer 且无 reducer，应用 VectorizeLoop
        - 如果推断出谓词条件，用 IfThenElse 包装循环

- `attr::kLayoutMap` 是一个 Block 注解属性，用于存储 buffer 到[布局](https://github.com/tile-ai/tilelang/blob/main/src/layout/layout.h) 的映射
  - 注解方式
    - 手动：`T.annotate_layout()`
    - 自动：通过 `LayoutInference` pass 推断
  - 分类
    - Fragment Layout：用于 `local.fragment` scope 的 buffer，表示数据在寄存器中的分布模式
      - makeGemmFragmentA/makeGemmFragmentACDNA：GEMM A 矩阵
      - makeGemmFragmentB：GEMM B 矩阵
      - makeGemmFragmentC/makeGemmFragmentCCDNA/makeGemmFragmentCHopper：GEMM 累加器 C 矩阵
      - makeGemmVoltaFragmentC/makeGemmVoltaFragmentA：Volta 架构专用
    - Shared Memory Layout：用于 `shared` 或 `shared.dyn` scope 的 buffer，优化 bank conflict 和内存访问模式
      - Linear Layout：不同于Triton的Linear，我认为是命名问题
        - makeGemmLayoutLinear：简单的行主序
        - makeGemmABLayoutPadded：带 padding 的布局，避免 bank conflict
      - Swizzle Layout：只支持了TMA所需的4种
        - makeFullBankSwizzleLayout：128B swizzle
        - makeHalfBankSwizzleLayout：64B swizzle
        - makeQuarterBankSwizzleLayout：32B swizzle
        ``` python
        using Swizzle128B = cute::Swizzle<3, 4, 3>;
        using Swizzle64B = cute::Swizzle<2, 4, 3>;
        using Swizzle32B = cute::Swizzle<1, 4, 3>;
        ```
    - Tensor Memory Layout：用于 `shared.tmem` scope 的 buffer，使用特殊的 tensor memory 布局以使用 `TCGEN5MMA` 指令
      - Layout D
      - Layout E
    - 架构特定
      - makeGemmABLayout：通用的 Tensor Core 布局
      - makeGemmABLayoutHopper：Hopper 架构（H100）的 WGMMA 布局
      - makeGemmABLayoutSm100：Blackwell 架构（B100）的 TCGEN5MMA 布局
      - makeGemmABLayoutCDNA：AMD CDNA 架构的布局
      - makeGemmVoltaABLayout：Volta 架构的布局
      - makeTensorOpMultiplicand：Tensor Core 操作数的通用布局
      - makeGemmSparseAmpereABLayout：Ampere 稀疏 GEMM 的布局
    
#### Pass 8：LowerTileOp

#### Pass 9：LowerL2Persistent

#### Pass 10：LegalizeVectorizedLoop

#### Pass 11：LegalizeSafeMemoryAccess

#### Pass 13：LoopVectorizeDynamic

### Phase-2: OptimizeForTargetHardware

- Phase-2根据目标硬件的能力选择不同的优化路径并注入相应的优化

```python
if have_tma(target) and not disable_warp_specialized and not disable_tma_lower:
    # Hopper 路径（H100/H200）
    # - TMA（Tensor Memory Accelerator）：bulk 异步复制
    # - Warp Specialization：Producer/Consumer 角色拆分
    # - WGMMA：4-warp 组矩阵乘加
    # - mbarrier：可计数字节屏障
else:
    # Ampere 路径（A100 及更早）
    # - cp.async：异步全局→共享拷贝
    # - 传统软件流水线
    # - commit_group/wait_group 同步
```

### Hopper 优化路径：TMA + Warp Specialization

#### Pass 1：LowerSharedBarrier &　Pass 2：LowerSharedTmem

 - `LowerSharedBarrier` 将用户声明的 barrier buffer（用于线程同步）转换为底层的 PTX barrier 初始化调用
   - 转换前
     ``` python
     data_is_ready = T.alloc_buffer((128,), "uint64", scope="shared.barrier")  
     compute_is_done = T.alloc_buffer((128,), "uint64", scope="shared.barrier")
     ```
   - 转换后
      ``` python
      data_is_ready = T.alloc_buffer((1,), "uint64", scope="shared")  
      compute_is_done = T.alloc_buffer((1,), "uint64", scope="shared")
      if tx == 0:  # 或使用 shuffle_elect  
        T.ptx_init_barrier_thread_count(data_is_ready[0], 128)  
        T.ptx_init_barrier_thread_count(compute_is_done[0], 128)
      ```
    - `mbarrier` 是共享内存对象，需要**编译期确定地址**以便后续引用

    - 实现在[src/transform/lower_shared_barrier.cc](https://github.com/tile-ai/tilelang/blob/main/src/transform/lower_shared_barrier.cc)

  - `LowerSharedTmem` 的作用与 LowerSharedBarrier 类似，但针对 tensor memory buffer，将用户声明的 `shared.tmem` buffer 转换为底层的初始化调用，使得 Blackwell 架构的 TCGEN5 指令可以正确访问这些 buffer

#### Pass 3：IfStmtBinding

  - `IfStmtBinding` 的主要作用是将单个 if 语句展开为多个 if 语句，使得 if 条件应用到 then 分支中的每个子语句上，为 `if` 语句绑定条件/域信息，确保在流水线规划和 buffer 分配之前，if 语句已经被正确展开
  - 转换前
    ``` python
    if condition:  
      stmt1  
      stmt2  
      stmt3
    ```
  - 转换后
    ``` python
    if condition:  
        stmt1  
    if condition:  
        stmt2  
    if condition:  
        stmt3
    ```

  - `IfStmtBinding` 和 `MergeIfStmt` 为互补Pass，在软件流水线注入过程中，`InjectSoftwarePipeline` 需要处理被 if 语句包装的循环体，`IfStmtBinding` 的展开使得流水线 pass 可以更细粒度地控制每个语句的执行条件，而 `MergeIfStmt` 则在流水线注入后清理冗余的 if 语句
    
#### Pass 4：MultiVersionBuffer

  - `MultiVersionBuffer` 将原本的 shared memory buffer 转换为多版本 buffer，在 shape 的第一维添加 **版本数（num_stages）**，使得流水线的不同阶段可以同时访问 buffer 的不同版本，实现内存加载 `copy(i+1)` 与计算 `compute(i)` 的重叠执行
    - 顺序 
      ```cpp
      // 单版本（顺序）
      __shared__ float A[128][64];
      for (int i = 0; i < N; ++i) {
        copy(global, A);
        __syncthreads();
        compute(A);
        __syncthreads();
      }
      ```
    - 3 阶段流水
      ```cpp
      __shared__ float A[3][128][64];  // stage ring
      for (int i = 0; i < N; ++i) {
        int stage = i % 3;
        copy(global, A[stage]);            // 与上一 tile 的 compute 并发
        compute(A[(stage - 1 + 3) % 3]);
      }
      ```

* **Pass 5：WarpSpecialized**

  * **位置**：`phase.py:129`
  * **注册**：`src/transform/warp_specialized_rewriter.cc:1292`
  * **作用**：将“同构线程执行”改写为“**Producer/Consumer 异构 warpgroup**”。
  * **传统 vs WS**（简化对比）：

    ```cpp
    // 传统：所有 warp 交替 copy/compute
    for (int i = 0; i < N; ++i) { copy(); __syncthreads(); compute(); __syncthreads(); }

    // WS：部分 warp 专职 copy，部分 warp 专职 compute
    if (warp_id < 4) { // Producer
      for (int i = 0; i < N; ++i) {
        mbarrier_expect_tx(&barrier[i%3], bytes);
        tma_load_async(global, shared[i%3]);
        mbarrier_arrive(&barrier[i%3]);
      }
    } else {           // Consumer
      for (int i = 0; i < N; ++i) {
        mbarrier_wait_parity(&barrier[i%3], i%2);
        wgmma(shared[i%3], acc);
      }
    }
    ```
  * **优势**：Producer/Consumer 真并发、降低寄存器压力、提升 occupancy 与局部性。
  * **改写步骤**：识别相位 → 角色标注 → 注入 `mbarrier` 协议 → 分裂控制流（按角色执行）。
  * **门控**：

    ```python
    configs = {"tl.disable_warp_specialized": True}
    ```

#### Pass 6：InjectTmaBarrier

  * **位置**：`phase.py:130`
  * **注册**：`src/transform/inject_tma_barrier.cc:526`
  * **作用**：按 TMA 传输字节数注入完整 `mbarrier` 协议（expect_tx / arrive / wait_parity）。
  * **协议示意**：

    ```cpp
    // Producer 相位
    mbarrier_expect_tx(&barrier[stage], N_BYTES);
    cp.async.bulk.tensor.shared::cluster.global.mbarrier::complete_tx::bytes [dst], [src], ...;
    mbarrier_arrive(&barrier[stage]);

    // Consumer 相位
    mbarrier_wait_parity(&barrier[stage], parity);
    wgmma(...);
    ```
  * **字节数计算**：例如 `128x64` 的 `fp16` → `128*64*2 = 16384` 字节。
  * **限制**：仅支持 `threadIdx.x` 线性维度；多维 block（如 `16x16`）会禁用并告警。
  * **门控**：

    ```python
    configs = {"tl.disable_tma_lower": True}  # 强制禁用 TMA，回退 cp.async
    ```

#### Pass 7：AnnotateWarpGroupRegAlloc

  - `AnnotateWarpGroupRegAlloc` 分析函数中的寄存器提示调用，并在 producer 和 consumer 分支中注入适当的 `set_max_nreg` 调用，降低溢出或提升并行度。

  - 实现
    1. 收集：使用 `SetMaxNRegCollector` 类收集函数中的寄存器提示
      - 识别 set_max_nreg() 调用，提取寄存器数量和增减标志，返回 `[dec_reg, inc_reg]` 数组
      - 识别 no_set_max_nreg() 调用，标记禁用寄存器限制，返回 `[-1, -1]
      - 检测是否已经存在自定义 warp specialization，返回 `[dec_reg, inc_reg]` 数组
    2. 检测：使用 `SimtCopyDetector` 类检测是否存在 SIMT 拷贝操作，通过检查 BufferStore 的目标 scope 来判断，如果存储到非 global scope，则认为有 SIMT copy
    3. 注入：使用 `SetMaxNRegInjector` 类注入
      - 条件
        - 必须有有效的寄存器提示（dec_reg >= 0 && inc_reg >= 0）
        - 不能有 SIMT copy 操作
      - 逻辑：识别 kWarpSpecializationScope 属性的 IfThenElse 结构
        - 为 producer 分支注入 dec_reg_stmt（减少寄存器），默认为 24 个寄存器
        - 为 consumer 分支注入 inc_reg_stmt（增加寄存器），默认为 240 个寄存器
    4. 生成：在 CodeGen 环节`set_max_nreg` 调用最终会被降低为 PTX 指令
      ``` cpp
        std::string func_name = is_inc ? "tl::warpgroup_reg_alloc" : "tl::warpgroup_reg_dealloc";  
        this->stream << func_name << "<" << std::to_string(nreg) << ">();\n";
      ```
      对应的 CUDA 模板实现
      ``` cpp
        template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_alloc() {  
          asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));  
        }  
          
        template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_dealloc() {  
          asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));  
        }
      ```

#### Pass 8: PipelinePlanning & Pass 9: InjectSoftwarePipeline**

  * 这两个 Pass 负责分析循环体并生成高效的流水线代码（**prologue / steady / epilogue**）

    ```python
    loop.attr["software_pipeline_stage"] = [0, 1, 2]
    loop.attr["software_pipeline_order"] = [0, 2, 1]
    loop.attr["software_pipeline_async_stages"] = 2
    ```

    ```cpp
    // 展开后（示意）
    // Prologue
    stage0(tile=0); stage0(tile=1);
    // Steady
    for (int i = 0; i < N-2; ++i) { stage1(tile=i); stage0(tile=i+2); }
    // Epilogue
    stage1(tile=N-2); stage1(tile=N-1);
    ```

  - `PipelinePlanning`是分析阶段，负责为循环体中的语句分配流水线阶段和执行顺序
    1. 语句分类与依赖分析：通过 `BufferRegionCollector` 分析每个语句的 buffer 访问模式
       - 识别 从 global 到 shared 的 copy 操作
       - 分析 buffer 的读写依赖关系
       - 构建异步依赖链（AsyncDependencyChainBuilder）
    2. 建立 Producer-Consumer：通过 CopyStageDependencyReadsManager 建立 copy 操作与其依赖的传递关系
       - 收集所有 copy 操作读取的 buffer
       - 标记为 copy 操作生产数据的语句为 producer_for_copy
       - 传递性地扩展依赖关系，直到收敛
    3. 最后使用分析：为每个 copy 操作确定其数据的最后使用位置
      ``` cpp
        // 分析 use-def 链确定 last_use_stmt_index  
        for (const BufferRegion &read : pipeline_stage_infos[i].reads) {  
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),  
                           [&](const BufferRegion &r) {  
                             return r->buffer == read->buffer &&  
                                    MayConflict(r->region, read->region);  
                           }) != pinfo.writes.end()) {  
            pinfo.last_use_stmt_index = std::max(pinfo.last_use_stmt_index, i);  
          }  
        }
      ```
    4. 阶段和顺序分配：基于依赖分析结果分配流水线阶段
       - Copy 操作分配到 stage 0（早期阶段）
       - 计算操作分配到 stage num_stages（后期阶段）
       - 根据 last_use_stmt_index 优化 copy 操作的调度位置
    5. 将分析结果以注解形式附加到循环上
       - software_pipeline_stage：每个语句的流水线阶段
       - software_pipeline_order：每个语句的执行顺序
       - software_pipeline_async_stages：异步操作的阶段（通常是 stage 0）
       - InjectSoftwarePipeline：流水线代码生成
       
  - `InjectSoftwarePipeline` 是代码生成阶段，将 `PipelinePlanning` 的注解转换为实际的流水线代码
    1. 识别带有流水线注解的循环
       ``` cpp
      bool HasPipelineAnnotation(const ForNode *op) const {  
        auto it1 = op->annotations.find(tir::attr::software_pipeline_stage);  
        auto it2 = op->annotations.find(tir::attr::software_pipeline_order);  
        return (it1 != op->annotations.end()) && (it2 != op->annotations.end());  
      }
    ```
    2. 分析 buffer 访问模式并计算所需版本数
      - 计算每个 buffer 的定义阶段（def）和最后使用阶段（use）
      - 版本数 = use - def + 1，确保流水线各阶段不会产生数据竞争
      - 调用 RewriteAllocBuffer 为需要版本化的 buffer 添加版本维度

    3. 生成 prologue、body、epilogue 三个部分
    ``` cpp
    // Prologue: 预热阶段，填充流水线  
    Stmt prologue = EmitImpl(pipeline_loop_->min, pipeline_loop_->min + max_stage_, true, true);  
    // Body: 稳态阶段，流水线满载运行  
    Stmt body = EmitImpl(pipeline_loop_->min + max_stage_, pipeline_loop_->min + pipeline_loop_->extent, false, false);  
    // Epilogue: 排空阶段，清空流水线  
    Stmt epilogue = EmitImpl(pipeline_loop_->min + pipeline_loop_->extent, pipeline_loop_->min + pipeline_loop_->extent + max_stage_, true, true);
    ```
    4. PipelineBodyRewriter 重写 buffer 访问，添加版本索引 `PrimExpr new_index = old_index + floormod(pipeline_loop_->loop_var, new_buffer->shape[0]) * offset;`，确保不同迭代访问 

#### 2.5 WGMMA 与控制流清理

* **Pass 10：LowerOpaqueBlock**

  * **位置**：`phase.py:138`
  * **注册**：`src/transform/lower_opaque_block.cc:236`
  * **作用**：显式降解不透明 Block，释放后续改写空间。

* **Pass 11：MergeIfStmt**

  * **位置**：`phase.py:139`
  * **注册**：`src/transform/merge_if_stmt.cc:103`
  * **作用**：合并相邻同条件 `if`，精简 CFG。

* **Pass 12：RewriteWgmmaSync**

  * **位置**：`phase.py:140`
  * **注册**：`src/transform/wgmma_sync_rewriter.cc:271`
  * **作用**：收集同域内 WGMMA 发射，按需插入 `warpgroup_wait<N>`，避免异步数据冒险。
  * **门控**：

    ```python
    configs = {"tl.disable_wgmma": True}  # 回退通用 MMA
    ```

#### 2.6 Fence Proxy — 异步写可见性保证

* **Pass 13：InjectFenceProxy**

  * **位置**：`phase.py:142`
  * **注册**：`src/transform/inject_fence_proxy.cc:199`
  * **作用**：在 **Async 通道写**（TMA/WGMMA）与 **Generic 通道读** 之间插入 `fence.proxy.async`，保证可见性。
  * **PTX 指令**：

    ```asm
    fence.proxy.async.shared::cta;
    ```

### 普通优化路径

### 3. 公共尾段优化

* **形态优化**：索引窄化（`NarrowDataType`）、`FlattenBuffer`、`VectorizeLoop`。
* **存储优化**：`MergeSharedMemoryAllocations`、`StorageRewrite`。
* **Host/Device 拆分**：`SplitHostDevice`；设备侧 Codegen → PTX/HIP；宿主侧包装与调用。
* **cp.async 注入（Ampere 关键）**：识别 Phase-1 谓词化模式，改写为 `cp.async + commit_group/wait_group`。
* **打包与启动**：统一调用接口（packed API）与适配器执行。

---

### 4. 总结

* **Phase-2 目标**：基于硬件能力注入特定优化。
* **Hopper 路径**：TMA + Warp Specialization + WGMMA + mbarrier + fence proxy。
* **Ampere 路径**：cp.async + 传统软件流水 + commit/wait 同步。
* **公共尾段**：完成形态/存储/向量化/Host-Device 拆分与最终注入。
* **关键 Pass**：`WarpSpecialized`、`InjectTmaBarrier`（Hopper）；`InjectPTXAsyncCopy`（Ampere）。

## Codegen
