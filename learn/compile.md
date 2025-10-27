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
    ``` python
    # 用户写的代码
    acc = T.alloc_fragment([8, 8], "float32")
    acc[0] = initial_value  # 孤立写入

    # Pass 改写后
    for _ in T.parallel(1):  # 包裹层
        acc[0] = initial_value
    ```

#### Pass 4：InjectAssumes

#### Pass 5 & 12：Simplify

#### Pass 6：LayoutReducer & Pass 7：LayoutInference

- 共同负责确定和配置 TileLang 中 fragment 和 shared memory 的内存布局

- `LayoutReducer` 根据访问模式和注解，为 `local.reducer` 类(如 `T.reduce`)配置布局属性（如何在 `threadIdx.x` 上复制/分片），随后 pipeline 会将其当作 `local.fragment` 使用/进一步 Lower，这些信息随后被 `LayoutInference` 使用来推断相关 buffer 的布局
    - Reducer 的布局需要考虑线程间的数据复制和归约模式，因此需要特殊布局处理

- `LayoutInference` 收集算子与循环的 use-def 关系，基于线程绑定与目标硬件，对 fragment / shared 的布局与并行循环的线程映射/谓词/向量化进行全局推断与改写，并把结果写回到 IR 注解与循环结构中，为后续 LowerTileOp、TMA/cp.async、向量化等优化做铺垫
    1. 收集：`BufferUseDefCollector::Collect`
        a. 扫描 Call（TileOp）与 For(kParallel)，构建推断对象列表 infer_list_、使用表 use_list_；
        b.
        c.
    2. `LayoutInferencer` 类将推断的布局应用到 IR 上

#### Pass 8：LowerTileOp

#### Pass 9：LowerL2Persistent

#### Pass 10：LegalizeVectorizedLoop

#### Pass 11：LegalizeSafeMemoryAccess

#### Pass 13：LoopVectorizeDynamic

### Phase-2: OptimizeForTargetHardware

### 1.1 核心分支：Hopper vs Ampere

* **位置**：`tilelang/engine/phase.py:120`
* **分支判断逻辑**：

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

* **关键能力判断函数**：

```python
from tilelang.contrib.nvcc import have_tma

# 通常对应 sm_90+（Hopper）
if have_tma(target):
    print("支持 TMA！")
```

---

### 2. Hopper 优化路径：TMA + Warp Specialization

#### 2.1 早期准备：共享资源初始化

* **Pass 1：LowerSharedBarrier**

  * **位置**：`phase.py:123`
  * **注册**：`src/transform/lower_shared_barrier.cc:209`
  * **作用**：在共享内存中创建并初始化 `mbarrier`（Barrier with Byte Counter），为 TMA 同步提供原语。
  * **核心 API（概念）**：

    ```cpp
    // 初始化
    mbarrier_init(&barrier, num_threads);
    // Producer 发起异步传输前声明期望字节数
    mbarrier_expect_tx(&barrier, bytes);
    // 异步传输（TMA 自动递减计数）
    tma_load_async(...);
    // Producer 到达
    mbarrier_arrive(&barrier);
    // Consumer 等待（0/1 交替）
    mbarrier_wait_parity(&barrier, parity);
    ```
  * **固定槽位原因**：`mbarrier` 是共享内存对象，需要**编译期确定地址**以便后续引用。
  * **实现示意**：

    ```cpp
    __shared__ alignas(8) uint64_t mbarrier[num_stages];
    if (threadIdx.x == 0) {
      for (int i = 0; i < num_stages; ++i) mbarrier_init(&mbarrier[i], threads_per_block);
    }
    __syncthreads();
    ```

* **Pass 2：LowerSharedTmem**

  * **位置**：`phase.py:125`
  * **注册**：`src/transform/lower_shared_tmem.cc:305`
  * **作用**：下沉 `shared.tmem` 的初始化与占位，为 WGMMA 的共享内存 staging。
  * **协同关系**：**TMA → shared.tmem → WGMMA**（由 `mbarrier` 进行同步）。

#### 2.2 Warp Specialization

* **Pass 3：IfStmtBinding**

  * **位置**：`phase.py:128`
  * **注册**：`src/transform/if_stmt_binding.cc:86`
  * **作用**：为 `if` 语句绑定条件/域信息，便于后续 WS 与流水线规划理解控制流角色。

* **Pass 4：MultiVersionBuffer**

  * **位置**：`phase.py:128`
  * **注册**：`src/transform/multi_version_buffer_rewriter.cc:331`
  * **作用**：构造**环形/多版本缓冲**，让 copy/compute **重叠**执行。
  * **单版本 vs 多版本**：

    ```cpp
    // 单版本（顺序）
    __shared__ float A[128][64];
    for (int i = 0; i < N; ++i) {
      copy(global, A);
      __syncthreads();
      compute(A);
      __syncthreads();
    }

    // 多版本（3 阶段流水）
    __shared__ float A[3][128][64];  // stage ring
    for (int i = 0; i < N; ++i) {
      int stage = i % 3;
      copy(global, A[stage]);            // 与上一 tile 的 compute 并发
      compute(A[(stage - 1 + 3) % 3]);
    }
    ```
  * **为何需要环形缓冲**：实现 `copy(i+1)` 与 `compute(i)` 并发，通常 2–5 个版本避免冲突。

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

#### 2.3 TMA Barrier 协议注入

* **Pass 6：InjectTmaBarrier**

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

#### 2.4 寄存器预算与流水线规划

* **Pass 7：AnnotateWarpGroupRegAlloc**

  * **位置**：`phase.py:131`
  * **注册**：`src/transform/annotate_warp_group_reg_alloc.cc:170`
  * **作用**：为不同角色注入寄存器预算（Producer 少、Consumer 多），降低溢出或提升并行度。

    ```cpp
    if (warp_role == Producer) asm volatile("setmaxnreg.sync %0;" :: "r"(32));
    if (warp_role == Consumer) asm volatile("setmaxnreg.sync %0;" :: "r"(128));
    ```

* **Pass 8–9：PipelinePlanning + InjectSoftwarePipeline**

  * **位置**：`phase.py:134-135`
  * **注册**：`src/transform/pipeline_planning.cc:710`；`src/transform/inject_pipeline.cc:1007`
  * **作用**：规划阶段/顺序/依赖并展开为 **prologue / steady / epilogue**。

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

#### 2.7 Hopper 路径小结

* 已创建并初始化 **mbarrier**（固定槽位）。
* 构造 **多版本共享缓冲**，实现 copy/compute 重叠。
* 改写为 **Warp Specialization**（Producer/Consumer）。
* 注入 **TMA 协议**（expect_tx / arrive / wait_parity）。
* 规划并展开 **软件流水线**（prologue/steady/epilogue）。
* 重写 **WGMMA 同步**（`warpgroup_wait<N>`）。
* 插入 **Fence Proxy** 保证异步写的可见性。
* **验证命令**：

  ```bash
  # 检查 TMA
  grep -E "tensormap|cp\.async\.bulk\.tensor" kernel.ptx
  # 检查 mbarrier
  grep -E "mbarrier\.(arrive|wait_parity|expect_tx)" kernel.ptx
  # 检查 WGMMA
  grep -E "wgmma\.mma_async|warpgroup_wait" kernel.ptx
  # 检查 Fence
  grep "fence.proxy.async" kernel.ptx
  ```

---

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