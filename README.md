# RYUKI —— 基于 MLIR 的算子融合编译器实践

> 一个从零搭建的 MLIR Out-of-Tree 项目，实现 Linalg 算子融合与完整 Lowering 流水线。

---

## 博客大纲：从 Memory Wall 到算子融合——用 MLIR 手写一个编译器优化 Pass

---

### 〇、Hook：一道让所有 AI 芯片工程师睡不着的数学题

> 一块 H100 GPU 的理论算力是 **989 TFLOPS（FP16）**，
> 显存带宽是 **3.35 TB/s**。
>
> 做一次 4096×4096 的矩阵乘法：
>   - 需要的计算量：**~137 GFLOP**
>   - 需要搬运的数据：**~192 MB**
>   - 计算"消耗"带宽的速度：**989 TFLOPS / 3.35 TB/s ≈ 295**
>
> 这意味着：芯片每搬 1 字节数据，可以做 295 次浮点运算。
> 而一个朴素的 Matmul+Bias 实现，实际算术强度只有 **~4 FLOP/Byte**。
>
> **295 倍的算力，被带宽活活饿死。**

这就是 Memory Wall（存储墙）——现代 AI 编译器需要解决的核心矛盾。

---

### 一、Memory Wall：AI 芯片的阿喀琉斯之踵

#### 1.1 冯·诺依曼瓶颈与算术强度

- **Roofline 模型**：以算术强度（FLOP/Byte）为横轴，展示算子的性能上界
- **计算受限 vs. 带宽受限**：为什么大多数 DNN 算子跑在 Roofline 的左侧
- **数据量化**：不同精度下访存开销的具体数字（FP32 / FP16 / INT8）

| 算子 | 算术强度（典型值） | 瓶颈 |
|------|------------------|------|
| Matmul（大矩阵）| ~N/2 FLOP/Byte | 计算受限 |
| Elementwise Add | ~0.25 FLOP/Byte | **带宽受限** |
| Matmul + Bias   | 未融合时同上 | **带宽受限** |

#### 1.2 中间张量的代价：一次被忽视的灾难

```
A[M,K] × B[K,N] → C[M,N]   ← 写一次：M×N×4 Byte
C[M,N] + bias[M,N] → out    ← 再读一次：M×N×4 Byte
```

对于 M=N=4096：
- 中间张量 C 占 **64 MB**
- 它在 HBM 和 L2 cache 之间完整地写一遍、再读一遍
- 实际有效计算量：Bias Add 只有 `2 × M × N ≈ 33M` FLOPs
- 算术强度：`33M / 64MB ≈ 0.5 FLOP/Byte` ← 极度带宽受限

#### 1.3 解法：算子融合（Op Fusion）

核心思想：**消灭中间张量，让数据留在寄存器/L1 Cache 里复用。**

融合后：
```
for i, j:                     ← 只写一次 out[i,j]
    acc = bias[i,j]           ← 以 bias 初始化累加器
    for k:
        acc += A[i,k] * B[k,j]
    out[i,j] = acc
```

理论加速比（带宽受限场景）：减少一次完整的 HBM 往返读写。

---

### 二、现代 AI 编译器的融合策略全景

#### 2.1 融合的三个层次

```
Level 3: 图融合（Graph-level）
  ├─ 框架层：TorchScript / TF Graph 的 Op 合并
  └─ 工具：torch.compile、XLA、TVM

Level 2: 调度融合（Schedule-level）
  ├─ 循环融合 + Tiling：将多个算子的循环合并
  └─ 工具：TVM Schedule、Halide

Level 1: IR 融合（IR-level）← 本文重点
  ├─ 在编译器 IR 层面匹配模式并重写
  └─ 工具：MLIR Pattern Rewrite、XLA HLO Fusion
```

#### 2.2 为什么选 MLIR？

- **多层次 IR**：同一套框架支持从 HLO 到 LLVM 的全栈表示
- **Dialect 机制**：每个 Dialect 是一个"语义域"，可以精确控制降级时机
- **Linalg Dialect**：专为线性代数算子设计，保留高层语义到 lowering 前夕
- **Pattern Rewrite**：声明式的模式匹配 + 变换，编写融合 Pass 只需 50 行 C++

#### 2.3 与 TVM / XLA 的横向对比

| 维度 | TVM | XLA | MLIR（本项目）|
|------|-----|-----|--------------|
| 融合粒度 | Schedule | HLO Pattern | IR Pattern |
| 扩展方式 | Python Schedule | C++ | C++ OOT |
| 调试难度 | 中 | 高 | 中（mlir-opt）|
| 生产就绪 | ✓ | ✓ | 框架（IREE）|

---

### 三、MLIR Linalg Dialect 设计哲学

#### 3.1 Tensor vs. MemRef：两个世界

```mlir
// Tensor 语义（函数式，不可变）← fusion 在这一层做
%C = linalg.matmul ins(%A, %B) outs(%zeros) -> tensor<MxNxf32>

// MemRef 语义（命令式，可原地修改）← bufferization 后
linalg.matmul ins(%A, %B : memref<...>) outs(%C : memref<...>)
```

**为什么融合要在 Tensor 层做？**
- 没有别名（Aliasing）问题，分析精确
- 数据流清晰，模式匹配简单
- 下游 bufferization 可以联合优化内存分配

#### 3.2 Named Op vs. linalg.generic

```
linalg.matmul             ← 具名结构化算子（保留高层语义）
    ↓ generalize
linalg.generic {          ← 统一表示（indexing_maps + region）
  indexing_maps = [...],
  iterator_types = [parallel, parallel, reduction]
} { ... }
```

**Structured Op 的核心**：每个算子由三要素完整描述：
1. **Indexing Maps**：描述每个操作数的访问模式（仿射变换）
2. **Iterator Types**：区分 parallel / reduction 维度
3. **Region Body**：标量级别的计算逻辑

#### 3.3 Destination-Passing Style（目标传递语义）

```mlir
// outs 不是"输出"，而是"初始值"
// 语义：result = outs_init + Σ(computation)
%result = linalg.matmul
    ins(%A, %B)
    outs(%init)   ← 提供初始值，result 是新 tensor
```

这个设计是融合技巧的理论基础：
- `outs(%zeros)` → 从零开始累加 → `result = A@B`
- `outs(%bias)`  → 从 bias 开始累加 → `result = bias + A@B` ✓

---

### 四、项目实现深度解析

#### 4.1 项目全局架构

```
.mlir 输入
  │
  ▼ [Stage 1] Tensor 级融合（本文核心）
  │   Pattern Rewrite: matmul + add → 单一 matmul
  │
  ▼ [Stage 2] Bufferization（One-Shot）
  │   Tensor → MemRef，别名分析，消除不必要拷贝
  │
  ▼ [Stage 3] Loop 层降级
  │   linalg.generic → affine.for → scf.for
  │
  ▼ [Stage 4] LLVM Dialect
      → LLVM IR → 汇编
```

#### 4.2 Pattern Rewrite 机制详解

MLIR 的 Pattern Rewrite 框架是一个**局部变换引擎**：

```
┌─────────────────────────────────────────────┐
│          GreedyPatternRewriteDriver          │
│                                              │
│  Worklist: [Op1, Op2, Op3, ...]              │
│      ↓                                       │
│  对每个 Op 尝试所有已注册的 Pattern            │
│      ↓                                       │
│  Pattern.match() → 成功？→ Pattern.rewrite() │
│      ↓                                       │
│  新产生的 Op 加回 Worklist                    │
│      ↓                                       │
│  重复直到不动点（没有 Op 被修改）              │
└─────────────────────────────────────────────┘
```

关键 API 对照：

```cpp
// 注册 Pattern
RewritePatternSet patterns(ctx);
patterns.add<MatmulBiasAsInitPattern>(ctx);

// 驱动器（贪心策略）
applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
```

#### 4.3 融合 Pattern 的匹配逻辑（核心代码解析）

**匹配条件（缺一不可）：**

```
条件 1: linalg.add 的某个输入由 linalg.matmul 产生
条件 2: matmul 的结果只有一个使用者（即这个 add）
         ↑ 否则不能消除中间张量 %C
条件 3: matmul 的 outs 初始值来自 linalg.fill(0.0)
         ↑ 否则替换语义会改变（bias-as-init 技巧的前提）
条件 4: matmul 输出类型与 bias 类型一致
```

**变换（3 步完成）：**

```cpp
// Step 1: 用 bias 替换 zeros，创建新 matmul
auto fusedMatmul = rewriter.create<linalg::MatmulOp>(
    loc, resultType,
    ValueRange{A, B},    // inputs 不变
    ValueRange{bias}     // outs: zeros → bias  ← 核心变换
);

// Step 2: 将 add 的所有结果引用改为指向新 matmul
rewriter.replaceOp(addOp, fusedMatmul.getResults());

// Step 3: 删除已无用的旧 matmul（fill 和 empty 由 DCE 清理）
rewriter.eraseOp(matmulOp);
```

#### 4.4 两种融合策略对比

| 策略 | 输出 IR | 代码复杂度 | 适用场景 |
|------|---------|-----------|---------|
| **A: Bias-as-Init** | `linalg.matmul(outs=bias)` | 低（约 60 行）| 生产代码 |
| **B: 显式 Generic** | `linalg.generic{...}` | 高（约 120 行）| 自定义 body、教学 |

策略 B 的 indexing maps 构建展示：

```cpp
// 3 个循环：i(parallel), j(parallel), k(reduction)
AffineExpr d0 = getAffineDimExpr(0); // i
AffineExpr d1 = getAffineDimExpr(1); // j
AffineExpr d2 = getAffineDimExpr(2); // k

SmallVector<AffineMap> maps = {
    AffineMap::get(3, 0, {d0, d2}), // A[i,k]
    AffineMap::get(3, 0, {d2, d1}), // B[k,j]
    AffineMap::get(3, 0, {d0, d1}), // out[i,j]
};
```

---

### 五、LLVM IR 层面的量化对比

#### 5.1 实验设置

- 输入：`matmul_bias(A: 4×16, B: 16×8, bias: 4×8)` —— f32
- 命令：`ryuki-runner input.mlir [--no-fusion] --emit-llvm`
- 统计口径：直接统计生成的 `.ll` 文件中各 LLVM IR 指令出现次数

#### 5.2 关键指令数量对比（实测）

| 指标 | 未融合 | 融合后 | 降低比例 |
|------|--------|--------|---------|
| 总 LLVM IR 行数 | 188 | 109 | **↓ 42%** |
| Load 指令数 | 5 | 3 | **↓ 40%** |
| Store 指令数 | 3 | 1 | **↓ 67%** |
| malloc（中间 buffer）| 1 | 0 | **↓ 100%** |
| 循环层数（for 嵌套）| 2×2 + 3 | 3 | **↓ 2 层** |
| 分支指令（br）| 21 | 9 | **↓ 57%** |

> 未融合版本需额外分配 `malloc(192B)` 的中间 buffer 存放 matmul 结果，再启动第二段双重循环做 bias add；融合后无中间 buffer，整个计算在单段三重循环中完成，bias 直接作为累加器初始值。

#### 5.3 关键 IR 片段对比（实测生成）

**融合前**（三段独立循环：fill → matmul → bias add，中间 buffer 需 malloc）：

```llvm
; ① malloc 中间 buffer（融合后消失）
%43 = call ptr @malloc(i64 192)

; ② fill 循环：将中间 buffer 清零（2层嵌套）
%57 = phi i64 ...          ; i = 0..4
  %61 = phi i64 ...        ; j = 0..8
    store float 0.0, ptr %67   ; 写 0 到中间 buffer

; ③ matmul 循环：A@B 累加到中间 buffer（3层嵌套）
%73 = phi i64 ...          ; i = 0..4
  %77 = phi i64 ...        ; j = 0..8
    %81 = phi i64 ...      ; k = 0..16
      %93 = load float, ptr %92   ; Load A[i,k]
      %103 = load float, ptr %102  ; Load B[k,j]
      %108 = load float, ptr %107  ; Load tmp[i,j]（每次迭代都读写中间 buffer！）
      %109 = fmul float %93, %103
      %110 = fadd float %108, %109
      store float %110, ptr %114  ; Store → 中间 buffer（HBM 往返）

; ④ bias add 循环：读中间 buffer + bias（2层嵌套）
%122 = phi i64 ...         ; i = 0..4
  %126 = phi i64 ...       ; j = 0..8
    %133 = load float, ptr %132  ; Load tmp[i,j]（刚写完又读！）
    %143 = load float, ptr %142  ; Load bias[i,j]
    %144 = fadd float %133, %143
    store float %144, ptr %148   ; Store → output
```

**融合后**（单段三重循环，无中间 buffer，bias 直接初始化累加器）：

```llvm
; 无 malloc！无 fill 循环！直接进入单段三重循环
%44 = phi i64 ...          ; i = 0..4
  %48 = phi i64 ...        ; j = 0..8
    %52 = phi i64 ...      ; k = 0..16
      %64 = load float, ptr %63   ; Load A[i,k]
      %74 = load float, ptr %73   ; Load B[k,j]
      %84 = load float, ptr %83   ; Load out[i,j]（即 bias，DPS 语义）
      %85 = fmul float %64, %74
      %86 = fadd float %84, %85   ; acc 在寄存器内累加
      store float %86, ptr %95    ; Store → out（直接写目标）
```

核心差异：融合前内层循环每次迭代需要从中间 buffer **Load + Store**（3 次内存操作 → HBM 往返）；融合后 bias 作为 outs 直接用 DPS 语义初始化，内层循环 Load 直接读累加器当前值，**消除了独立的 fill 阶段和 malloc**。

#### 5.4 理论加速分析

对 M=N=K=4096 的生产规模：

```
中间 tensor 大小：4096 × 4096 × 4 Byte = 64 MB

未融合：
  Store C[M,N] → HBM：64 MB
  Load  C[M,N] ← HBM：64 MB
  总额外带宽：128 MB

融合后：
  C 仅活在寄存器（L1 Cache）中
  额外带宽：≈ 0

HBM 带宽节省：128 MB / (3.35 TB/s) ≈ 38 µs
（对 ~1ms 的 Matmul 来说：约 3.8% 的端到端加速，
 但在 Memory-Bound 场景（小批量 batch）可达 10-30%）
```

---

### 六、工程经验总结与踩坑实录

#### 6.1 三个让我卡了最久的坑

**坑 1：isZeroFill 的必要性**
```
错误：直接把 add.outs 替换到 matmul.outs
后果：如果 matmul 原本 outs 是非零值（如 GEMM 的 alpha/beta 变换），
      替换后语义变为 bias + non_zero_init + A@B，结果错误
正确：严格检查 matmul.outs 来自 fill(0.0)
```

**坑 2：hasOneUse() 的约束**
```
错误：不检查 matmul 结果的使用次数
后果：如果 %C 被两个 add 使用，删除 matmul 后第二个 add 引用悬空
正确：hasOneUse() 确保 %C 只有一个消费者
```

**坑 3：Bufferization 前的融合时机**
```
错误：在 bufferization 之后再做融合
后果：MemRef 有别名问题，isZeroFill 匹配失败（fill 变成 memset）
正确：所有 Pattern Rewrite 融合必须在 bufferize 之前完成
      这是 MLIR 一类约定俗成的不成文规则
```

#### 6.2 MLIR 开发工具链心得

```bash
# 最重要的调试命令：逐阶段看 IR
ryuki-runner input.mlir --stop-after=fusion    -o - | less
ryuki-runner input.mlir --stop-after=bufferize -o - | less
ryuki-runner input.mlir --stop-after=affine    -o - | less

# 每 Pass 后验证（报错精确定位）
ryuki-runner input.mlir --verify-each

# 用 mlir-opt 单独测试 Pattern
mlir-opt --ryuki-matmul-bias-fusion input.mlir

# DEBUG 日志（需要 -DLLVM_ENABLE_ASSERTIONS=ON）
ryuki-runner input.mlir --debug-only=ryuki-matmul-bias-fusion
```

---

### 七、延伸与展望

#### 7.1 本项目的局限

- 只处理 f32，未处理混合精度（f16 + f32 accumulator）
- 未做 Tiling：大矩阵时 bias 初始化的 cache 效率仍有改进空间
- 未处理 batch matmul（linalg.batch_matmul）

#### 7.2 生产级融合系统的做法

| 系统 | 融合方式 | 关键技术 |
|------|---------|---------|
| IREE | Linalg Fusion + Dispatch Region | 根据 target 做 tiling-then-fusion |
| TVM Relax | StructuralPattern + BlockBuilder | 图级 + 算子级联合优化 |
| XLA | HLO Fusion | 贪心 producer-consumer 合并 |
| Triton | Python DSL | 直接表达 fused kernel |

#### 7.3 值得继续探索的方向

1. **Vectorization**：在 affine 层加入 `--affine-super-vectorize`，生成 SIMD 指令
2. **Tiling + Fusion**：`linalg-tile-and-fuse-tensor-ops` 做带 tiling 的融合
3. **GPU Backend**：将 Lowering 目标改为 GPU Dialect → NVVM → PTX
4. **自动调优**：结合 MLIR 的 Transform Dialect 做 schedule 搜索

---

### 八、结语

> 算子融合本质上是在回答一个问题：
> **"计算和数据，谁先移动？"**
>
> 未融合的编译器让数据在内存层级之间反复搬运；
> 融合后的编译器让数据留在计算单元附近，让计算追上带宽。
>
> 从 Memory Wall 出发，到 MLIR 的 Pattern Rewrite，再到 LLVM IR 的指令数变化——
> 这条路展示了一个系统级优化是如何从数学直觉，一步步落地成可执行代码的。

---

## 快速上手

```bash
# 1. 克隆 & 构建（需要已安装 LLVM/MLIR）
git clone <this-repo> && cd RYUKI
mkdir build && cd build
cmake .. -G Ninja \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
ninja -j$(nproc)

# 2. 运行融合 Pass，查看效果
./bin/ryuki-runner ../test/Toy/matmul_bias.mlir --stop-after=fusion -o -

# 3. 导出 LLVM IR
./bin/ryuki-runner ../test/Toy/matmul_bias.mlir --emit-llvm -o output.ll
cat output.ll
```

详细构建与使用说明见 [BUILD.md](BUILD.md)。
