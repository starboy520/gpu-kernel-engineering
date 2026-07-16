# GPU Kernel Engineering

面向 GPU 性能工程的 CUDA Kernel 作品集。项目从手写 Kernel 出发，使用 correctness、Compute Sanitizer、CUDA Event benchmark、Nsight Compute 和 SASS 建立可复现的证据链。当前正式结果采集于 NVIDIA A100 80GB PCIe（`sm_80`）。

## 代码归属与 AI 协作边界

本仓库采用“学习者实现核心 Kernel，AI coding agent 搭建工程与验证框架”的协作方式。

### 学习者亲自实现

- CUDA Kernel 的核心计算逻辑；
- thread/Warp/CTA 与数据的映射；
- Shared Memory 与寄存器布局；
- reduction、Online Softmax、同步与 pipeline；
- `cp.async`、Warp shuffle 以及后续 `mma.sync/ldmatrix` 核心路径；
- 根据 benchmark、ncu 和 SASS 证据提出优化假设。

### AI coding agent 负责

- CMake、公共头文件和项目脚手架；
- CPU/cuBLAS reference、runner、命令行接口和 validation；
- correctness matrix、CTest 与 Compute Sanitizer 脚本；
- CUDA Event benchmark、Nsight Compute、SASS 提取和结果渲染；
- 文档整理、失败复现、代码 review 和分级调试提示。

除非学习者明确要求某个具体修改，AI coding agent 不替代学习者编写核心 Kernel body。

## 四个子项目

| 子项目 | 状态 | 核心目标 | 入口 |
| --- | :---: | --- | --- |
| 1. FP32 GEMM Optimization | ✅ 已完成 | CUDA Core 上从 Naive 到 Vectorized/Async 的优化阶梯 | [projects/gemm](projects/gemm/) |
| 2. FP32 FlashAttention Baseline | ✅ 已完成 | 从 Materialized Attention 重建 Online Tiled 与异步 K/V 数据流 | [projects/flash_attention](projects/flash_attention/) |
| 3. Advanced Attention Prefill | ✅ M1/M2 已完成 | `Br=4` Query tiling 与 Warp-per-query ownership | [projects/attention_prefill](projects/attention_prefill/) |
| 4. Tensor Core GEMM | 🔄 G1 框架就绪 | G0～G5：WMMA、`mma.sync`、`ldmatrix`、最小 GEMM、Warp Tile | [projects/gemm_tensorcore](projects/gemm_tensorcore/) |

四个子项目均已有独立入口。Tensor Core GEMM 当前只有 reference、runner、测试和 G1 Kernel TODO，核心 WMMA 逻辑尚未实现。

---

## 子项目 1：FP32 GEMM Optimization

在 row-major `C=A×B` 上完成 CUDA Core 优化阶梯。

| 阶段 | 核心变化 | 状态与证据 |
| --- | --- | --- |
| G-V0 Naive | 每个 thread 计算一个输出元素 | correctness 基线 |
| G-V1 Shared Tiling | A/B Tile 进入 Shared Memory 复用 | `2048³` 为 Naive 的 1.46× |
| G-V2 Register Tiling | 每个 thread 计算 `8×4` 输出 | 建立 thread-level output tile |
| G-V3 Vectorized | `float4` 128-bit Global load | **12.88 TFLOPS，73.0% pedantic FP32 cuBLAS** |
| G-V4 Async 16B | `cp.async` 双缓冲 | long scoreboard 下降，但 Shared 瓶颈导致墙钟回退约 4.8% |
| G-V5 Evidence | canonical benchmark、ncu、SASS | `LDG.E.128` 与 `LDGSTS.E.BYPASS.128` 已验证 |

Async 负结果被保留：它证明“生成异步指令、降低某项 stall”不等于最终墙钟一定更快。

- [完整优化阶梯](projects/gemm/README.md)
- [A100 canonical 结果](projects/gemm/results/generated/a100-fp32.md)
- [实验方法](projects/gemm/docs/methodology.md)
- [Vectorized SASS](projects/gemm/results/evidence/vectorized-sass.md)
- [Async SASS](projects/gemm/results/evidence/async-16b-sass.md)

---

## 子项目 2：FP32 FlashAttention Baseline

当前范围为单 batch、单 head、FP32、forward-only educational/research baseline。

| 里程碑 | 实现内容 | 状态与证据 |
| --- | --- | --- |
| F0 Reference | CPU double reference、统一 runner | 完成 |
| F1 Naive Materialized | `QK^T → Stable Softmax → PV`，显式 `N×N` workspace | 完成 |
| F2 Online Tiled | `Bc=16`，running `m/l/O_acc`，不物化 `N×N` | 完成 |
| F3 Warp Reduction | Warp shuffle 并行 max/sum | correctness/safety 通过，稳定墙钟收益不足，保留负结果 |
| F4 Async K/V | `cp.async` 双缓冲 | 硬件指令已生成，收益依赖 Shape |
| F5 Evidence | 48 行 benchmark、ncu、SASS | 完成 |

代表结果：non-causal `D=128` 下 Async 在 `N=512` 回退 18.2%，`N=768` 近似持平，在 `N=1024` 提升 17.6%。因此不宣称跨 Shape 稳定加速。

- [项目说明与里程碑](projects/flash_attention/README.md)
- [Naive Kernel](projects/flash_attention/kernels/naive.cu)
- [Online Tiled Kernel](projects/flash_attention/kernels/tiled.cu)
- [Warp Reduction Kernel](projects/flash_attention/kernels/tiled_parallel.cu)
- [`cp.async` Kernel](projects/flash_attention/kernels/tiled_async.cu)
- [A100 canonical benchmark](projects/flash_attention/results/generated/a100-fp32.md)

---

## 子项目 3：Advanced Attention Prefill

该项目在冻结的 FP32 FlashAttention baseline 上继续研究 Query Tile 与 Warp ownership。

### M1：`Br=4` Query-tiled FP32 SIMT

```text
一个 CTA 处理最多 4 条 Query
→ 只加载一份 K/V Tile
→ 四条 Query 复用 K/V
→ 每行独立维护 m/l/alpha/O_acc
```

完成步骤：

1. 建立 `Q[4,D]`、`K/V[16,D]`、`Score[4,16]` 数据流；
2. 处理 Query、K/V、feature 三种 tail；
3. causal 使用全局 Query/Key 坐标；
4. 完成 126 个 Shape + 2 个特殊输入；
5. 完成四类 sanitizer；
6. 对比 `Br=1` 与 `Br=4` 的 CUDA Event、ncu 和 SASS。

M1 结果：理论 K/V requested elements 减少 4×；在当前 A100/FP32 映射下，`N≤512` 回退，`N≥1024` 获益，最大为 `2048×64 causal=1` 的 1.489×。

### M2：Warp-per-query FP32 SIMT

```text
CTA = 4 Warps
Warp 0/1/2/3 → Query row 0/1/2/3
Warp lanes 合作 QK reduction 与 Online Softmax
Q/O_acc fragment 保存在 Lane 私有寄存器
```

完成步骤：

1. 每个 Warp 固定拥有一条 Query；
2. Lane 分片持有 Q feature；
3. Warp 合作计算每个 Score，并由 Lane 0～15 保存 Key Score/Weight；
4. Warp shuffle 完成 max/sum/broadcast；
5. 每个 Lane 使用寄存器 `O_acc`，沿 Key 维独立累加；
6. 复用 M1 的 128-case correctness 与 sanitizer；
7. 对比 M1/M2 的 CUDA Event、registers、Shared Memory、stall 与 SASS。

M2 开发期证据：non-causal `D=128` 提升约 5.2%～27.8%；`1024×128` 下 barrier stall 下降约 96.5%，Shared bank conflict 被消除，SASS 出现 `SHFL` 且无 `LDL/STL` spill。当前主要待优化点是 causal 路径中的未来 Tile/Key 无效工作。

- [项目说明](projects/attention_prefill/README.md)
- [M1/M2 Roadmap](projects/attention_prefill/ROADMAP.md)
- [M1 canonical benchmark](projects/attention_prefill/results/generated/a100-fp32-m1.md)
- [M1 ncu](projects/attention_prefill/results/evidence/m1-ncu-summary.md)
- [M2 开发期总结](projects/attention_prefill/results/evidence/m2-development-summary.md)

后续里程碑暂定：M3 FP16/BF16 SIMT、M4 MMA Playground、M5 Tensor Core QK、M6 Tensor Core PV。

---

## 子项目 4：Tensor Core GEMM（G0～G5）

该子项目与 FP32 GEMM 分离，固定为 FP16 input、FP32 accumulator/output，并采用独立 reference、容差与结果表。

| 阶段 | 学习与实践 | 验收 |
| --- | --- | --- |
| G0 | FP16 数值语义、`m16n8k16`、Warp collective、手算 | Shape/FLOP/寄存器账本 |
| G1 | WMMA 单 Warp `16×16×16` | ones/identity/random、`HMMA` |
| G2 | 单条 `mma.sync.m16n8k16` | Lane/Register dump 与 logical rebuild |
| G3 | `ldmatrix.x1/x2/x4/trans` | 映射表、sanitizer、`LDSM` |
| G4 | `ldmatrix + mma.sync`，单 Warp `16×8×K` | K=16/32/48、`LDSM/HMMA` |
| G5 | `16×32`、`32×32` Warp Tile | registers、spill、occupancy、候选 Tile |

G6 Multi-Warp、G7 `cp.async` Multi-stage、G8 swizzle/tail/cuBLAS 正式结果在 G5 后再决定。若目标优先转向 Tensor Core Attention，可在 G5 后进入 Attention M4/M5/M6。

- [G0～G5 执行 Roadmap](projects/gemm/TENSOR_CORE_ROADMAP.md)
- [Tensor Core GEMM 项目入口](projects/gemm_tensorcore/README.md)

---

## 统一性能工程流程

每个实现型里程碑遵循同一顺序：

1. **Math & Ownership**：先画数学、Shape 和线程/数据所有权；
2. **Correctness**：CPU 或 vendor reference，对拍最小输入和边界 Shape；
3. **Safety**：memcheck、racecheck、synccheck、initcheck；
4. **Wall-clock**：CUDA Event，固定 warmup/iterations/repeats，保存中位数与 spread；
5. **Nsight Compute**：验证资源、stall、cache、吞吐和瓶颈假设；
6. **SASS**：确认目标机器指令、向量宽度、shuffle、MMA、异步复制和 spill；
7. **Conclusion**：同时保留收益、回退和 inconclusive 结果。

性能结论必须绑定 GPU、Shape、dtype、实现路径和测量协议。ncu duration 不替代正常 CUDA Event 墙钟，静态 SASS 数量不代表 runtime 执行次数。

## 项目结构

```text
common/                       公共 validation、runner 与 shell 工具
projects/
├── gemm/                     FP32 CUDA Core GEMM
├── flash_attention/          FP32 FlashAttention baseline
├── attention_prefill/        Advanced Attention Prefill M1/M2
└── gemm_tensorcore/          Tensor Core GEMM G0～G5
```

## 构建与复现

需要 CUDA Toolkit、CMake 3.25+；ncu/SASS 证据还需要 Nsight Compute CLI 与 `cuobjdump`。项目在 VS Code 中通过 CMake Tools 构建，并通过 CTest 运行自动化测试。

各子项目的正式入口：

```bash
projects/gemm/scripts/validate.sh
projects/gemm/scripts/sanitize.sh full
projects/gemm/scripts/benchmark.sh

projects/flash_attention/scripts/validate.sh
projects/flash_attention/scripts/sanitize.sh full
projects/flash_attention/scripts/benchmark.sh

projects/attention_prefill/tests/test_query_tiled.sh
projects/attention_prefill/tests/test_warp_per_query.sh
projects/attention_prefill/scripts/sanitize.sh full
```

具体 Shape、输入模式、输出位置和 profiler 命令以各子项目 README 为准。
