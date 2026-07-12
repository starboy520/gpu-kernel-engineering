# A100 Tensor Core GEMM 学习与实现路线

本文面向已经完成 CUDA Core FP32 GEMM 优化、但尚未手写过 Tensor Core 指令的开发者。目标不是复制一个能运行的 kernel，而是按“先理解指令，再理解数据布局，最后建立流水线”的顺序，独立完成 A100 上的 Tensor Core GEMM。

> [!IMPORTANT]
> 当前 FP32 CUDA Core 首版已经完成。Tensor Core 是独立的 v2 路线，不替换现有结果，也不与 pedantic FP32 数据混在同一张性能表中。在真正实现前，本文只记录学习计划和验收标准，不声明尚未测得的性能。

## 1. 最终学习目标

完成这条路线后，应当能够不依赖模板回答并验证以下问题：

1. 一个 warp 如何共同执行一次矩阵乘加，而不是每个 thread 各算一个独立输出。
2. `mma.sync` 的 `m16n8k16` 分别描述什么，为什么它不是 thread-block tile。
3. A、B、C fragment 分别由每个 lane 的哪些寄存器共同组成。
4. `ldmatrix` 如何把 shared-memory tile 分发到 32 个 lane 的寄存器。
5. 为什么 shared-memory layout 必须同时服务于 global store、`ldmatrix` 和 bank conflict。
6. 如何从 instruction tile 扩展为 warp tile，再扩展为 block tile。
7. 如何用 `cp.async` 建立 global → shared → register → Tensor Core 的多级流水线。
8. 如何在 SASS 中证明生成了 Tensor Core 和矩阵加载指令。
9. 为什么 FP16、BF16、TF32 和 pedantic FP32 必须使用不同的 reference、误差阈值和 baseline。

最终作品不是“调用了一个 Tensor Core API”，而是一条可以解释的证据链：

```text
数值语义
  ↓
warp / lane 映射
  ↓
shared-memory layout
  ↓
ldmatrix + mma.sync
  ↓
block tiling + pipeline
  ↓
correctness + sanitizer
  ↓
wall-clock + ncu + SASS
```

## 2. 已有知识如何迁移

现有 FP32 GEMM 并不会推倒重学。大部分概念可以直接迁移，只是粒度发生了变化。

| 已掌握的 CUDA Core 概念 | Tensor Core 中的对应概念 |
| --- | --- |
| 每个 thread 计算一个或多个 C 元素 | 一个 warp 共同持有并更新 C fragment |
| thread-level register tiling | warp-level MMA tiling |
| block tile | 由多个 warp tile 组成的 thread-block tile |
| `float4` global load | 向量化 global → shared 搬运 |
| shared-memory tiling | 为 `ldmatrix` 设计的 shared-memory tile |
| 双缓冲与 `cp.async` | 多 stage global → shared pipeline |
| `FFMA` SASS | `HMMA` Tensor Core SASS |
| shared load 指令 | `LDSM` 矩阵加载 SASS |

最大的思维变化是：**CUDA Core GEMM 的基本计算者通常是 thread，Tensor Core GEMM 的基本计算者是 warp。**

## 3. 先区分三层接口

Tensor Core 不是单一 API。学习时要分清抽象层次，避免会用 WMMA 却不理解最终指令。

| 层次 | 典型形式 | 适合做什么 | 局限 |
| --- | --- | --- | --- |
| C++ API | `nvcuda::wmma` | 第一次理解 fragment 和 warp collective | fragment 内部布局不透明，控制较少 |
| PTX ISA | `mma.sync`、`ldmatrix` | 学习 lane/register 映射并手写核心路径 | 需要自己管理寄存器和 shared layout |
| SASS | `HMMA`、`LDSM` | 验证机器代码与分析执行行为 | 不适合作为主要编程接口 |

推荐顺序是：

```text
WMMA 建立直觉
  ↓
单条 mma.sync 建立寄存器模型
  ↓
ldmatrix 建立 lane / shared-layout 模型
  ↓
组合成手写 GEMM
  ↓
用 SASS 验证
```

WMMA 是学习台阶，不是最终目标；PTX 是本阶段需要真正掌握的层级；SASS 用于确认和解释结果。

## 4. 先选择 FP16 路线，不从 TF32 开始

A100 支持多种 Tensor Core 输入。第一条手写路线推荐采用：

```text
FP16 A × FP16 B + FP32 C → FP32 D
```

对应的核心 PTX 形状为：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

选择 FP16 的原因：

- `m16n8k16` 是 Ampere 上非常典型的 MMA 形状；
- FP16 数据能直接配合 `ldmatrix` 的 16-bit 矩阵加载；
- 可以完整学习 `ldmatrix + mma.sync + cp.async` 主线；
- FP32 accumulator 保留了混合精度 GEMM 最常见的累加方式；
- 后续学习 BF16 时，整体结构可以复用。

TF32 放在 FP16 路线之后。TF32 的价值是保持 FP32 输入和指数范围，但乘法精度降低，并且常用的 `m16n8k8` 操作数组织与 FP16 路线不同。过早从 TF32 开始会把“数值转换”“寄存器映射”和“MMA 基础”三个问题混在一起。

### 4.1 三种候选语义

| 路线 | A/B 存储 | 乘法输入精度 | 累加 | 第一阶段是否采用 |
| --- | --- | --- | --- | --- |
| FP16 Tensor Core | FP16 | FP16 | FP32 | 是 |
| BF16 Tensor Core | BF16 | BF16 | FP32 | FP16 完成后 |
| TF32 Tensor Core | FP32，进入 MMA 前按 TF32 语义转换 | TF32 | FP32 | FP16 完成后 |

> [!WARNING]
> “FP32 accumulator”不等于“完整 FP32 GEMM”。只要乘法输入被量化为 FP16、BF16 或 TF32，结果就不能与当前 pedantic FP32 kernel 使用完全相同的数值声明。

## 5. 读懂一条 `mma.sync`

以下指令描述一次 warp-level 矩阵乘加：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

可以从左到右拆解：

| 字段 | 含义 |
| --- | --- |
| `mma` | 矩阵乘加，计算 D = A × B + C |
| `sync` | warp 内参与执行的线程同步完成该操作 |
| `aligned` | 所有参与线程执行相同的 MMA 指令和限定符 |
| `m16n8k16` | A 为 16×16，B 为 16×8，C/D 为 16×8 |
| `row.col` | A 采用 row-major 解释，B 采用 column-major 解释 |
| 第一个 `f32` | D 的元素类型 |
| 两个 `f16` | A、B 的元素类型 |
| 最后一个 `f32` | C accumulator 的元素类型 |

它执行的标量乘加数量为：

$$
16 \times 8 \times 16 = 2048\ \text{FMA}
$$

若一次 FMA 按两个浮点运算计数，则一条 warp-level MMA 对应：

$$
2 \times 16 \times 8 \times 16 = 4096\ \text{FLOP}
$$

### 5.1 它不是一个 thread 的工作

整个 warp 共同提供 A、B、C 操作数，也共同接收 D。对于 FP16 输入、FP32 累加的 `m16n8k16`：

| Fragment | 逻辑元素总数 | 每个 lane 持有 | 常见寄存器表示 |
| --- | ---: | ---: | --- |
| A：16×16 FP16 | 256 | 8 个 FP16 | 4 个 32-bit packed 寄存器 |
| B：16×8 FP16 | 128 | 4 个 FP16 | 2 个 32-bit packed 寄存器 |
| C：16×8 FP32 | 128 | 4 个 FP32 | 4 个 FP32 寄存器 |
| D：16×8 FP32 | 128 | 4 个 FP32 | 4 个 FP32 寄存器 |

这里的“每个 lane 持有”不代表它只计算对应的四个输出。Tensor Core 内部完成跨 lane 的矩阵运算，lane 只负责提供和接收 fragment 的一部分。

### 5.2 必须建立的安全规则

- 32 个 lane 必须执行同一条 MMA 指令；
- MMA 不能只放在部分 lane 执行的分支中；
- warp 内使用的形状、layout 和数据类型必须一致；
- fragment 寄存器顺序必须严格匹配 PTX ISA 给出的映射；
- 不要根据“看起来连续”猜测 lane 到矩阵元素的关系；
- 跨架构时重新核对目标架构支持的 shape 和类型。

第一遍学习不要求背下完整 lane 映射，但必须能够对照 PTX ISA 图，把一个输出元素定位到对应 lane 和 accumulator 寄存器。

## 6. 理解 `ldmatrix`

`mma.sync` 消费的是分布在 warp 各 lane 寄存器中的 fragment。`ldmatrix` 的职责是让一个 warp 从 shared memory 协作加载矩阵，并按照 MMA 需要的方式把数据分发到寄存器。

FP16 路线最常见的形式包括：

```text
ldmatrix.sync.aligned.m8n8.x1.shared.b16
ldmatrix.sync.aligned.m8n8.x2.shared.b16
ldmatrix.sync.aligned.m8n8.x4.shared.b16
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
```

| 字段 | 含义 |
| --- | --- |
| `m8n8` | 基础加载单元是 8×8 矩阵 |
| `x1/x2/x4` | 一次加载 1、2 或 4 个 8×8 矩阵 |
| `b16` | 每个矩阵元素为 16 bit |
| `trans` | 按转置形式分发，常用于准备 B fragment |
| `shared` | 源数据位于 shared memory |

### 6.1 为什么它比普通 shared load 难

普通 load 通常可以按“一个 thread 一个地址”理解。`ldmatrix` 是 warp collective：

- 一部分 lane 提供 shared-memory 行地址；
- 整个 warp 参与执行；
- 加载的数据被重新分发到全部 lane；
- 每个目标寄存器通常打包两个 FP16；
- 地址布局、转置方式和寄存器顺序必须与后续 MMA 对齐。

因此，第一次接触时不要直接把它塞进完整 GEMM。先做一个独立的 `ldmatrix` 映射实验。

### 6.2 必做的映射实验

在 shared memory 中写入容易辨认的值，例如让每个元素编码自己的逻辑坐标：

```text
value(row, col) = row * 100 + col
```

然后执行一次 `ldmatrix`，把每个 lane 得到的 packed 寄存器写回 global memory。最后打印：

```text
lane 0: reg0=(...), reg1=(...)
lane 1: reg0=(...), reg1=(...)
...
lane 31: ...
```

这个实验需要分别回答：

1. 哪些 lane 提供行地址；
2. `x1`、`x2`、`x4` 分别产生多少目标寄存器；
3. 一个 32-bit 寄存器内两个 FP16 的顺序是什么；
4. 加上 `trans` 后 lane/register 映射如何变化；
5. shared-memory 行跨度改变后，是否仍满足对齐并得到相同逻辑矩阵。

> [!TIP]
> 第一次实现时，让每个逻辑 8 元素 FP16 行从至少 16-byte 对齐的位置开始。先消除对齐变量，再研究更复杂的 padding 或 swizzle。

## 7. 四级 tile 层次

Tensor Core GEMM 至少要同时区分四种 tile。名称混用是最常见的设计错误之一。

| 层次 | 示例 | 谁负责 | 作用 |
| --- | --- | --- | --- |
| Instruction tile | 16×8×16 | 一个 warp 的一条 MMA | 硬件基本操作 |
| Warp tile | 32×32×16 | 一个 warp 的多条 MMA | 提高 accumulator 复用 |
| Block tile | 128×128×32 | 多个 warp | global/shared 数据复用 |
| Grid problem | M×N×K | 全部 thread block | 完整 GEMM |

例如，一个 warp 要计算 32×32 的 C tile，使用 `m16n8k16` 时，在一个 K slice 内需要：

$$
\frac{32}{16} \times \frac{32}{8} = 2 \times 4 = 8
$$

组 accumulator fragment，也就是八次 MMA 才覆盖该 warp tile。K 每前进 16，又要对这八组 accumulator 各更新一次。

### 7.1 先画 tile，再写下标

在实现任何多 MMA kernel 前，先在纸上写出：

```text
block tile: BM × BN × BK
warp arrangement: WARPS_M × WARPS_N
warp tile: WM × WN
instruction tile: 16 × 8 × 16
```

并验证：

$$
BM = WARPS_M \times WM
$$

$$
BN = WARPS_N \times WN
$$

以及 `WM` 能被 16 整除、`WN` 能被 8 整除、K 主循环能按 16 前进。尾块暂时通过 launcher fallback 处理，不要在第一个 fast path 内同时解决所有边界。

## 8. 推荐学习阶段

不要一开始就实现 128×128 多 stage GEMM。下面每一阶段只增加一个新变量。

### 阶段 0：建立术语和数值语义

**学习内容：**

- Tensor Core、WMMA、MMA、fragment 的关系；
- FP16 输入、FP32 accumulate 的含义；
- `m16n8k16` 的矩阵尺寸和 FLOP 计数；
- warp collective 的一致执行要求。

**通过标准：**

- 能画出 A=16×16、B=16×8、C/D=16×8；
- 能解释为什么一条指令是 4096 FLOP；
- 能说明 FP32 accumulate 为什么仍不等于 pedantic FP32 GEMM。

### 阶段 1：WMMA 单 warp、单 tile

**只实现：**

- 一个 warp；
- 一个 16×16×16 WMMA tile；
- M/N/K 全部整除；
- FP16 A/B，FP32 C；
- 不做 shared swizzle、向量化或异步流水线。

**学习重点：**

- fragment 是 warp 共同对象；
- `load_matrix_sync`、`mma_sync`、`store_matrix_sync` 的职责；
- 所有 lane 必须以一致参数调用 warp-level API。

**通过标准：**

- identity matrix、全 1、小随机矩阵全部通过；
- 能在反汇编中找到 Tensor Core 指令；
- 能解释 WMMA 的 16×16 输出如何对应到底层多个 16×8 MMA 操作。

**此阶段不追求：** 性能、尾块、通用 shape。

### 阶段 2：隔离一条 `mma.sync`

**只实现：**

- 手动准备 A/B packed 寄存器；
- C accumulator 清零；
- 执行一条 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`；
- 将每个 lane 的四个 accumulator 写回并重建逻辑矩阵。

建议使用容易推导的输入：

| A | B | 预期结果 |
| --- | --- | --- |
| 全 1 | 全 1 | 每个 D 元素为 16 |
| 单位矩阵 | 编号矩阵 | D 等于对应 B tile |
| 只有一行非零 | 小整数矩阵 | 只有对应输出行非零 |

**通过标准：**

- 能指出每个输出元素位于哪个 lane 的哪个 accumulator 寄存器；
- 无需复制外部 kernel，也能根据 PTX 映射完成 D 写回；
- SASS 中出现预期的 `HMMA` 指令。

### 阶段 3：隔离 `ldmatrix`

**只实现：**

- global → shared 使用普通协作 load；
- shared → registers 使用 `ldmatrix`；
- 将寄存器内容直接 dump 回 host；
- 分别测试普通加载和 `trans` 加载。

**通过标准：**

- 32 个 lane 的输出与手工映射表一致；
- `compute-sanitizer` 不报告 shared-memory 地址问题；
- SASS 中出现 `LDSM`；
- 能解释为什么 B 的逻辑 layout 需要不同的加载/转置处理。

### 阶段 4：组合成最小手写 MMA GEMM

**只实现：**

- 一个 warp 计算一个 16×8 输出 tile；
- K 可为 16 的多倍数；
- global → shared → `ldmatrix` → `mma.sync` → global；
- M/N 只接受完整 tile；
- 非法 shape 暂时拒绝或由 launcher fallback。

**通过标准：**

- K=16、32、48 分别通过，证明 accumulator 能跨 K tile 累加；
- identity、全 1、随机数据通过；
- racecheck 和 synccheck 通过；
- 能根据输出错误形态区分 A 映射、B 转置和 accumulator 写回问题。

### 阶段 5：扩展为 warp tile

**新增一个变量：** 一个 warp 持有多组 accumulator，例如 32×32 输出。

**学习重点：**

- 一份 A fragment 服务多个 N 方向 MMA；
- 一份 B fragment 服务多个 M 方向 MMA；
- accumulator 数量与 register pressure 的关系；
- warp tile 变大后 ILP、occupancy 和数据复用的权衡。

**通过标准：**

- 先画出每组 MMA 对应的 C 子块；
- 所有 accumulator 组都能正确写回；
- 用编译器资源报告记录 registers/thread，不凭感觉选择 tile。

### 阶段 6：扩展为多 warp block tile

**新增内容：**

- 多个 warp 协作加载同一个 block tile；
- 每个 warp 负责不同的 M/N 子块；
- thread block 级同步；
- shared-memory 容量与 warp 排布设计。

候选参数必须通过实际编译和 profile 决定，不预先承诺最终值。可以从以下设计表开始比较：

| 候选项 | 需要观察的问题 |
| --- | --- |
| warp tile 32×32 | accumulator 较少，但 A/B fragment 复用有限 |
| warp tile 64×32 | M 方向复用增加，register pressure 上升 |
| warp tile 32×64 | N 方向复用增加，B fragment 和 accumulator 增多 |
| block tile 128×128 | 数据复用高，但 shared memory 和并行度压力更大 |

**通过标准：**

- 至少两个 block tile 的大矩阵正确；
- block 边界和 warp 子块无重叠、无遗漏；
- 性能优于最小单 warp 版本，而不是只增加代码复杂度。

### 阶段 7：加入 `cp.async` 多 stage pipeline

现有 FP32 项目已经学习过 `cp.async`，但 Tensor Core 路线要重新设计消费者：

```text
global A/B
  ↓ cp.async
shared stage s
  ↓ ldmatrix
register fragments
  ↓ mma.sync
accumulator registers
```

推荐先做两个 stage，再考虑更多 stage。每次迭代要明确：

- 当前 stage 正在被 `ldmatrix` 消费；
- 下一 stage 正在由 `cp.async` 填充；
- commit/wait 的组数是否正确；
- shared stage 复用前是否已经完成消费；
- padding 或 swizzle 是否破坏 16-byte async-copy 对齐。

**通过标准：**

- K=16、32、48 和更长 K 循环均正确；
- racecheck、synccheck 通过；
- SASS 同时出现 128-bit global-to-shared copy、`LDSM` 和 `HMMA`；
- wall-clock 改善后才能称为优化，只有 stall 指标下降不算完成。

### 阶段 8：边界、swizzle 与正式结果

最后再加入：

- M/N/K 尾块策略；
- fast path 与 fallback；
- shared-memory swizzle；
- FP16 cuBLAS Tensor Core baseline；
- 固定协议 benchmark；
- ncu 与 SASS 证据。

边界处理优先采用 launcher 级 fast-path/fallback 分离。不要在第一个 Tensor Core fast path 中混合大量逐元素分支，导致所有 warp 为少数尾块付费。

## 9. 正确性设计

Tensor Core 路线不能直接复用当前 FP32 输入和阈值，然后把误差变大解释为“混合精度正常”。必须先定义 reference 的数值语义。

### 9.1 推荐 reference

对于 FP16 A/B、FP32 accumulator：

1. host 生成 FP32 随机数；
2. 显式转换为 FP16，作为 kernel 的真实输入；
3. CPU reference 读取已经量化后的 FP16 值；
4. 转换回 FP32 后进行 FP32 累加；
5. 另设 `cublasGemmEx` baseline，A/B 为 `CUDA_R_16F`，C 为 `CUDA_R_32F`；
6. CPU reference 用于小 shape 定位错误，cuBLAS 用于大 shape 对拍与性能基线。

这样可以分开两类误差：

- **输入量化误差：** FP32 生成值转成 FP16 时已经发生；
- **累加顺序误差：** CPU、手写 kernel 和 cuBLAS 的归约顺序不同。

### 9.2 不要立即固定一个万能阈值

误差通常随 K、输入分布和数值尺度变化。正式阈值应通过以下用例建立：

| 用例 | 目的 |
| --- | --- |
| 全 0 | 检查初始化和无效 fragment |
| identity | 检查 layout 与写回 |
| 全 1 | 检查 K 累加次数 |
| 小整数 | 便于精确手算 |
| 固定 seed 随机数 | 回归测试 |
| 正负抵消输入 | 暴露累加误差 |
| 大 K | 观察误差随归约长度增长 |

阈值需要记录推导依据，并按数据类型单独配置。不要静默放宽现有 FP32 的 `atol/rtol`。

### 9.3 第一批 shape

| Shape | 检查目标 |
| --- | --- |
| 16×8×16 | 单条 MMA |
| 16×8×32 | 两轮 K 累加 |
| 16×8×48 | 三轮 K 累加与 stage 复用 |
| 32×32×16 | 多 MMA warp tile |
| 128×128×32 | 多 warp、单 block tile |
| 256×256×64 | 多 block 与多轮 K |
| 非整除 shape | fast path 拒绝或 fallback 是否正确 |

## 10. 常见错误与定位顺序

| 现象 | 优先怀疑 |
| --- | --- |
| 全部输出都是 0 | accumulator 未写回、MMA 未执行、输入 fragment 为 0 |
| 每个值都相差固定倍数 | K 循环次数、packed FP16 内容、重复 MMA |
| 8 行或 8 列为周期错位 | `ldmatrix` 子矩阵顺序或 `trans` 使用错误 |
| 数值正确但矩阵块位置交换 | accumulator lane/register 到 C 的写回映射错误 |
| K=16 正确，K=32 失败 | accumulator 被清零、shared stage 提前覆盖、同步错误 |
| 单 warp 正确，多 warp 失败 | warp tile 偏移、shared 写入冲突、block 同步错误 |
| 只有大矩阵失败 | grid/block tile 偏移、资源限制、stage 循环错误 |
| sanitizer 通过但结果随机 | warp 分支不一致、wait/commit 次序、未初始化 fragment |
| 出现 `HMMA` 但性能很低 | tile 太小、数据供应不足、occupancy 或 shared layout 问题 |

推荐调试顺序固定为：

```text
单条 mma.sync
  ↓
单独 ldmatrix dump
  ↓
单 warp单 K tile
  ↓
单 warp多 K tile
  ↓
多 MMA warp tile
  ↓
多 warp block tile
  ↓
多 stage pipeline
  ↓
尾块与 fallback
```

不要在多 warp、多 stage、大矩阵同时失败时直接盯着 ncu 指标。

## 11. 性能证据怎么建立

### 11.1 Wall-clock

吞吐仍按有效 GEMM 工作量计算：

$$
\text{GFLOPS} = \frac{2MNK}{t_{ms} \times 10^6}
$$

Tensor Core 结果需要单独表明：

- A/B/C 数据类型；
- accumulator 类型；
- cuBLAS compute type 和 math mode；
- fast path 的 shape/alignment 要求；
- 是否包含输入类型转换时间；
- 是否只测 GEMM kernel。

### 11.2 ncu

第一轮只回答四类问题：

1. **Tensor Core 是否忙：** tensor pipe 活跃度和 MMA 指令数量；
2. **数据是否供得上：** global/shared throughput、long/short scoreboard；
3. **资源是否限制并行：** registers/thread、shared memory/block、achieved occupancy；
4. **layout 是否有代价：** shared bank conflict、无效或重复 load。

不同 Nsight Compute 版本的 metric 名称可能变化。先通过当前版本的 metric 查询或 section 集合确认名称，不把网上某个版本的 metric 字符串直接写死到正式脚本中。

### 11.3 SASS

A100 FP16 Tensor Core 路线至少要寻找两类证据：

| 目标 | 常见 SASS 线索 | 证明什么 |
| --- | --- | --- |
| Matrix multiply-accumulate | `HMMA` | PTX MMA 落到了 Tensor Core 指令 |
| Matrix shared load | `LDSM` | shared → fragment 使用了矩阵加载路径 |

加入异步流水线后，再检查 global-to-shared 异步复制对应的机器指令。静态指令出现只证明“编译器生成了它”，不能单独证明执行效率或墙钟收益。

## 12. 建议的项目边界

当前 runner 的核心接口使用 FP32 A/B/C 指针。FP16 Tensor Core 会改变输入存储类型，因此实现时不要用强制类型转换硬塞进现有 `LaunchFn`。

推荐采用独立但相邻的 v2 runner：

```text
现有 gemm_runner
  └─ FP32 CUDA Core + pedantic FP32 baseline

规划 gemm_tensor_runner
  └─ FP16/BF16/TF32 Tensor Core + 对应 cuBLAS baseline
```

两条路线可以复用：

- shape 与命令行参数设计；
- CUDA Event 计时框架；
- CSV/render 脚本思想；
- sanitizer 与 profile 工作流；
- 结果证据格式。

但必须分开的内容包括：

- device buffer 数据类型；
- reference 数值语义；
- 容差；
- kernel registry；
- cuBLAS compute type；
- benchmark 结果表。

建议的规划名称如下，真正实现时再加入构建系统：

| 阶段 | 规划 kernel 名称 | 状态 |
| --- | --- | --- |
| WMMA 学习基线 | `wmma-fp16` | 未实现 |
| 单 warp 手写 MMA | `mma-fp16-warp` | 未实现 |
| 多 warp tiled | `mma-fp16-tiled` | 未实现 |
| 异步多 stage | `mma-fp16-pipeline` | 未实现 |
| Vendor baseline | `cublas-fp16-tc` | 未实现 |

## 13. 八个学习单元

| 单元 | 阅读与实验 | 交付物 |
| --- | --- | --- |
| 1 | WMMA、fragment、混合精度语义 | 16×16×16 WMMA 正确性实验 |
| 2 | PTX `mma.sync` shape 和寄存器映射 | 单条 MMA + lane accumulator dump |
| 3 | PTX `ldmatrix` 与 shared 地址 | `x1/x2/x4/trans` 映射表 |
| 4 | 组合单 warp GEMM | 16×8×K 手写 kernel |
| 5 | 多 MMA warp tile | 至少两个 warp-tile 候选与资源记录 |
| 6 | 多 warp block tile | 正确的多 block tiled GEMM |
| 7 | `cp.async` 多 stage | 正确性、sanitizer、SASS 证据 |
| 8 | baseline 与正式实验 | 独立 Tensor Core 结果表和结论 |

每个单元都遵循同一验收顺序：

```text
手算小例子
  ↓
CPU reference
  ↓
cuBLAS reference
  ↓
memcheck / racecheck / synccheck
  ↓
wall-clock benchmark
  ↓
ncu
  ↓
SASS
```

## 14. 面试时应能讲清楚的主线

完成后，不要只说“用了 Tensor Core，所以更快”。更有价值的叙述是：

1. 先完成 CUDA Core FP32 阶梯，建立 correctness、benchmark、ncu 和 SASS 方法；
2. 将计算粒度从 thread-level FMA 提升为 warp-level MMA；
3. 先用 WMMA 验证数值语义，再下沉到 `mma.sync` 和 `ldmatrix`；
4. 通过 lane dump 独立验证 fragment 映射，而不是复制神秘常量；
5. 从 instruction tile 推导 warp tile 和 block tile；
6. 用 `cp.async` 让 global load 与 Tensor Core 计算重叠；
7. 对比 wall-clock、Tensor pipe、shared conflict、occupancy 和 SASS；
8. 将 FP16/TF32 与 pedantic FP32 分表，避免用不同精度制造虚假加速比。

一句话记忆：

> **`mma.sync` 决定怎么算，`ldmatrix` 决定数据怎样进入 warp，tiling 和 pipeline 决定 Tensor Core 能否持续吃饱。**

## 15. 官方资料阅读顺序

优先阅读 NVIDIA 官方资料，不先依赖二手 kernel：

1. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：先读 WMMA、warp execution 和 shared memory 相关章节；
2. [PTX ISA：Warp-level matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)：逐字段核对 `mma.sync` 的 shape、类型和 fragment 映射；
3. [PTX ISA：Warp-level matrix load](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix)：核对 `ldmatrix` 地址、`x1/x2/x4` 和 `trans`；
4. [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)：确认 `cublasGemmEx` 的输入、输出、compute type 和 math mode；
5. [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)：在自己完成最小 `mma.sync` 和 `ldmatrix` 实验后，再用它理解 production GEMM 的层级和 pipeline。

阅读 PTX ISA 时，以项目实际目标 `sm_80` 为准。网上针对 Turing、Ampere、Hopper 的示例不能在没有核对 shape、类型和架构要求时混用。
