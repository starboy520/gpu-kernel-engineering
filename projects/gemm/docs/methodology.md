# GEMM 性能实验方法

这套流程的目标不是制造更多临时数据，而是让 A100 上已经得到的 benchmark、Nsight Compute 和 SASS 证据可以重复生成。所有命令都从仓库根目录执行，脚本根据自身位置定位仓库；默认二进制是 `build/projects/gemm/gemm_runner`。

## 环境与固定协议

当前证据环境为 NVIDIA A100 80GB（`sm_80`）、CUDA 13.3、Nsight Compute CLI 2026.2。正式 benchmark 使用固定 seed `1234`，每个 shape 预热 10 次、计时 50 次、独立重复 3 轮并取中位数。profile 使用 validate 模式、`warmup=0`，只采集匹配目标函数的一次 launch；默认 shape 为 `2048×2048×2048`。

工具版本、GPU 型号和 Git commit 都属于实验条件。跨机器、跨编译器或跨 profiler 版本的数据只能作为线索，不能直接拼进同一张对比表。

## 先正确，再谈性能

每次改变实现后按下面顺序验收：

```bash
cmake --build build -j
projects/gemm/scripts/validate.sh
projects/gemm/scripts/sanitize.sh quick
projects/gemm/scripts/sanitize.sh full
projects/gemm/scripts/benchmark.sh
projects/gemm/scripts/profile.sh vectorized
projects/gemm/scripts/profile.sh async-16b
projects/gemm/scripts/extract_sass.sh vectorized
projects/gemm/scripts/extract_sass.sh async-16b
```

`validate.sh` 负责完整 CTest 和 shape/path 对拍；`sanitize.sh` 检查越界、竞态、同步和未初始化访问；`benchmark.sh` 才是 wall-clock 延迟的正式来源。profile 和 SASS 负责解释结果，不替代正确性与 benchmark。

只检查脚本命令结构时可以显式传小 shape：

```bash
projects/gemm/scripts/profile.sh vectorized 128 128 128
```

需要使用非默认 runner 时，通过环境变量覆盖：

```bash
GEMM_RUNNER=/path/to/gemm_runner \
    projects/gemm/scripts/profile.sh vectorized 2048 2048 2048
```

SASS 提取脚本的可选二进制参数是位置参数：

```bash
projects/gemm/scripts/extract_sass.sh vectorized /path/to/gemm_runner
```

完整 ncu report 和文本输出位于 `projects/gemm/results/profiles/`，完整目标函数 SASS 位于 `projects/gemm/results/sass/`。这些产物体积或机器相关性较强，默认不进入 Git。`projects/gemm/results/evidence/` 只保存脚本生成的小型静态计数与关键指令片段。

## 怎么读 profiler 数字

Nsight Compute 会锁定 GPU 时钟，并为了采集不同 counter 对同一次 kernel 做多轮 replay。profile 命令的总耗时和报告中的 replay 时间都不是 wall-clock benchmark，不能拿来计算线上吞吐。只有在相同 GPU、构建、shape、metric 集合和采集协议下，才能横向比较 profiler 指标；延迟结论始终回到 `benchmark.sh` 的中位数。

当前 `2048³` 的关键对照已经测得：

| 指标 | Async 16B | Vectorized |
| --- | ---: | ---: |
| Long scoreboard | 0.05 | 1.88 |
| Short scoreboard | 1.87 | 0.49 |
| Shared load conflict | 2.2-way | 1.3-way |
| Achieved occupancy | 38.49% | 26.78% |
| Registers/thread | 58 | 83 |

这组证据说明 `cp.async` 确实隐藏了大部分 global-memory dependency，但 short-scoreboard 和 shared bank conflict 同时上升。完整推导见 [`float4` Vectorized 阶段](vectorized.md) 与 [`cp.async` 双缓冲阶段](async-pipeline.md)。

## 四问证据模板

每一轮性能结论都回答四个问题：

1. **假设是什么？** 明确要消除的瓶颈，例如 global-load dependency。
2. **只改了什么？** 说明单变量改动，避免同时调整 tile、layout 和 pipeline。
3. **指标是否支持？** 同协议比较 registers、occupancy、throughput、stall、bank conflict 和 SASS。
4. **wall-clock 是否改善？** 用正式 benchmark 中位数确认 profiler 改善是否转化为端到端收益。

Async 16B 是需要保留的负结果：128-bit `LDGSTS` 已生成，long scoreboard 也显著下降，但正式 benchmark 的墙钟延迟仍比 Vectorized 增加约 4.8%。这不是失败数据，不应删除或只展示局部利好。shared-memory swizzle 需要同时满足 16B async-copy 对齐和新的 bank 映射，首版明确延后；若继续实验，应先只改变 A 的 shared layout，再按同一协议重新回答上面的四个问题。

## 正式 A100 数据集

首版正式结果由 `projects/gemm/scripts/benchmark.sh` 在干净工作树上生成，测试代码 commit 为：

```text
505f7895e34585d3b0daac24e2fa245f624b4890
```

环境与协议：

```text
GPU：NVIDIA A100 80GB PCIe
CUDA / nvcc：13.3 / 13.3.33
shape：512³、1024³、2048³
warmup：10
iterations：50
repeats：3
统计量：每个 kernel + shape 的 latency 中位数
seed：1234
```

canonical 数据保存在 `projects/gemm/results/raw/a100-fp32.csv`，展示表由脚本生成到 `projects/gemm/results/generated/a100-fp32.md`。CSV 中包含实际执行路径、reference 来源、误差、工具链和 Git commit；对外引用性能数字时以 canonical CSV 为准。