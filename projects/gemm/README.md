# FP32 GEMM 优化阶梯

**状态：已完成首版，可复现。**

在 NVIDIA A100（`sm_80`）上实现 row-major FP32 GEMM：`C = A × B`。项目关注 CUDA Core 上的优化过程，以及 correctness、wall-clock benchmark、ncu 和 SASS 之间能否互相印证。

## 正式结果：2048³

| Kernel | Path | Latency (ms) | GFLOPS | % cuBLAS |
| --- | --- | ---: | ---: | ---: |
| Naive | `naive` | 4.697866 | 3656.951563 | 20.7% |
| Shared tiled | `shared` | 3.222119 | 5331.855120 | 30.2% |
| Register tiled | `register` | 2.644419 | 6496.652581 | 36.8% |
| Vectorized | `fast-float4` | 1.333719 | 12881.175770 | 73.0% |
| Async 16B | `fast-pipeline-16b` | 1.398333 | 12285.960382 | 69.6% |
| cuBLAS FP32 | `cublas-pedantic-fp32` | 0.973251 | 17652.051322 | 100.0% |

数据来自 A100 80GB PCIe，测试代码 commit `505f789`；cuBLAS 使用 pedantic FP32。每个 kernel 先 warmup 10 次、再计时 50 次，独立重复 3 轮后取 latency 中位数。

[完整 512³ / 1024³ / 2048³ 结果表](results/generated/a100-fp32.md) · [canonical raw CSV](results/raw/a100-fp32.csv) · [结果字段说明](results/README.md)

## 优化阶梯

下表的阶段结果均指正式 `2048³` 数据；它们不是对所有 shape 都成立的单调结论。

| 阶段 | 核心变化 | 阶段结果 |
| --- | --- | --- |
| [Naive](docs/naive.md) | 每个 thread 计算一个输出 | 3.66 TFLOPS，基线 |
| [Shared memory tiling](docs/shared-tiled.md) | A/B tile 数据复用 | 1.46× Naive |
| [2D register tiling](docs/register-tiled.md) | 每个 thread 计算 `8×4` 输出 | 1.22× Shared |
| [`float4` Vectorized](docs/vectorized.md) | 128-bit global load | 1.98× Register，12.88 TFLOPS |
| [`cp.async` 双缓冲](docs/async-pipeline.md) | 16B global-to-shared async copy | 0.95× Vectorized，负收益 |
| [cuBLAS baseline](docs/cublas-baseline.md) | pedantic FP32 reference | Vectorized 达到 73.0% |

`512³` 和 `1024³` 上的排序并不一致：例如 `512³` 的 Register 版本慢于 Shared，`1024³` 两者基本持平。小规模问题更容易受 launch、tile/grid 数量和并行度影响，因此这里不宣称优化阶梯对每个 shape 都单调加速，完整对照以 [三组 shape 正式表](results/generated/a100-fp32.md) 为准。

Vectorized 的 SASS 中有 14 条静态 `LDG.E.128`；Async 16B 则生成 4 条 `LDGSTS.E.BYPASS.128`。这说明宽加载与异步搬运确实落到了最终指令，但“指令生成”不等于“墙钟更快”。

- [Vectorized SASS 静态证据](results/evidence/vectorized-sass.md)
- [Async 16B SASS 静态证据](results/evidence/async-16b-sass.md)

## Async 负结果

在同一 `2048³` profile 协议下，Async 16B 相比 Vectorized：long scoreboard 从 `1.88` 降到 `0.05`，但 short scoreboard 从 `0.49` 升到 `1.87`，shared load bank conflict 从 `1.3-way` 升到 `2.2-way`。`cp.async` 隐藏了大部分 global-memory dependency，瓶颈却转移到 shared-memory 访问，最终 Async 为 12.29 TFLOPS，仅为 cuBLAS 的 69.6%，比 Vectorized 慢约 4.7%。

首版没有加入 swizzle。直接 padding 会破坏后续 shared row 的 16B async-copy alignment；正确处理需要成对修改 shared-memory 写入和读取映射。这个实验保留为可解释的负结果，不用局部 profiler 改善替代最终 wall-clock 结论。详见 [Async 阶段分析](docs/async-pipeline.md) 和 [实验方法](docs/methodology.md)。

## Build

需要 CUDA Toolkit、CMake 3.25+，默认目标架构为 `sm_80`。以下命令从仓库根目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 验收与复现

每个 kernel 共用同一套 runner、reference 和计时框架，并按 [CUDA GEMM Kernel 验收手册](docs/kernel-verification-guide.md) 执行。正式数据采集与 profiler 解读规则见 [GEMM 性能实验方法](docs/methodology.md)。

```bash
projects/gemm/scripts/validate.sh
projects/gemm/scripts/sanitize.sh quick
projects/gemm/scripts/sanitize.sh full
projects/gemm/scripts/benchmark.sh

projects/gemm/scripts/profile.sh vectorized 2048 2048 2048
projects/gemm/scripts/profile.sh async-16b 2048 2048 2048
projects/gemm/scripts/extract_sass.sh vectorized
projects/gemm/scripts/extract_sass.sh async-16b
```

<details>
<summary>自动化命令与输出约定</summary>

`validate.sh` 先运行完整 CTest，再按 `tests/correctness_cases.csv` 对拍五个手写实现及具名 fast path / fallback。`sanitize.sh quick` 运行代表性 memcheck；`full` 进一步运行 racecheck、synccheck 和 initcheck。

`benchmark.sh` 是正式 benchmark 的唯一入口。默认 shape 为 `512³`、`1024³`、`2048³`，协议为 warmup 10、iterations 50、repeats 3、seed 1234。脚本要求 clean working tree，并把中位数写入 `results/raw/a100-fp32.csv`，再生成 `results/generated/a100-fp32.md`。

非正式烟雾测试会自动写入 smoke 文件：

```bash
GEMM_SHAPES='512x512x512' \
GEMM_WARMUP=2 \
GEMM_ITERATIONS=3 \
GEMM_REPEATS=1 \
projects/gemm/scripts/benchmark.sh
```

各脚本默认使用 `build/projects/gemm/gemm_runner`。validate、sanitize、benchmark 可接收自定义 runner 位置参数；profile 通过 `GEMM_RUNNER` 覆盖。profile 的 `.ncu-rep` / `.txt` 与完整 `.sass` 是机器相关诊断产物，紧凑证据保存在 `results/evidence/`。

</details>

## Fast Path 与范围

Vectorized 与 Async 16B fast path 要求 A/B 基地址 16B 对齐，并满足 `K % 4 == 0`、`N % 4 == 0`；M 可以有尾块。不满足条件时，launcher 整体 fallback 到 Register Tiling，而不是在同一个 kernel 内混合 scalar/vector path。

首版不包含 Tensor Core、转置输入和 batched GEMM，也没有实现 shared-memory swizzle。当前结果只针对仓库记录的 A100 `sm_80` 环境与三组方阵 shape；跨 GPU、工具链或协议的数据不直接比较。
