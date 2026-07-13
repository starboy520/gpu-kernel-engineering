# FlashAttention 性能实验方法

## 证据分层

本项目将性能证据严格分为四层：

| 层级 | 工具 | 回答的问题 | 不能替代 |
| --- | --- | --- | --- |
| 正确性与安全 | CPU double reference、CTest、Compute Sanitizer | 结果和同步是否可信 | 性能结论 |
| 正常墙钟 | CUDA Event benchmark | 实际运行谁更快、快多少 | 瓶颈原因 |
| 微架构指标 | Nsight Compute | stall、occupancy、吞吐如何变化 | 正常墙钟 |
| 机器指令 | `cuobjdump` / SASS | 源码意图是否落成指令 | 动态执行次数和最终性能 |

> **记忆：Benchmark 判断结果，ncu 解释机制，SASS 验证编译。**

## Canonical benchmark 协议

正式结果统一由 `projects/flash_attention/scripts/benchmark.sh` 生成：

```text
GPU：NVIDIA A100 80GB PCIe（sm_80）
dtype：FP32
batch / head：1 / 1
layout：row-major
N：512、768、1024
D：64、128
causal：0、1
kernel：naive、tiled、tiled-parallel、tiled-async
input pattern：random
seed：1234
warmup：10
iterations：50
repeats：3
统计量：三次 latency 中位数
计时：CUDA Event
reference：CPU double
```

矩阵共有 $3\times2\times2=12$ 个 problem，四个 Kernel 共生成 48 个 canonical 行。`N=512/768/1024` 分别用于观察 Async 回退端、收益交界和收益端；causal 与 non-causal 分开报告，不计算混合平均值。

每次 runner 进程先完成 CPU reference 和 GPU correctness，再执行 warmup 和 CUDA Event 计时。Host 分配、H2D/D2H、CPU reference 和首次 correctness launch 不在 Event 区间内。Naive 的 Event 区间包含其 `QK^T → Softmax → PV` 三个 Kernel，因此代表完整 Naive 路径。

## 结果统计

每个 `(kernel,N,D,causal)` 独立运行三次，记录：

- `latency_ms`：三次中位数；
- `latency_min_ms` 与 `latency_max_ms`；
- `spread_pct = 100 × (max-min)/median`；
- 实际 `path`、误差、workspace、GPU、工具链、Git commit 和 runner SHA256。

Async 相对 Tiled 的定义：

$$
S_{async}=\frac{T_{tiled}}{T_{async}}
$$

$$
\Delta_{async}=100\left(\frac{T_{tiled}}{T_{async}}-1\right)
$$

`Async Δ` 为正表示改善，为负表示回退。分类规则：

- `benefit`：$S_{async}\ge1.05$ 且 Async spread 不超过 3%；
- `regression`：$S_{async}\le0.95$ 且 Async spread 不超过 3%；
- `near-parity`：差异小于 5% 且 spread 不超过 3%；
- `inconclusive`：spread 超过 3%。

三次中位数不是统计显著性证明，因此边界附近的小差异必须与 spread 一起展示。

## Clean-tree 与输出

Canonical 模式要求干净工作树，并记录：

- Git commit；
- runner SHA256；
- GPU 名称、UUID、compute capability 和 driver；
- CUDA/nvcc；
- build preset、seed 和计时协议。

正式矩阵全部成功后，脚本才原子替换结果文件：

- raw CSV：`projects/flash_attention/results/raw/a100-fp32.csv`；
- generated table：`projects/flash_attention/results/generated/a100-fp32.md`。

修改 shape、warmup、iterations 或 repeats 时自动进入 smoke 输出，不能覆盖 canonical 数据。工作树不干净时，smoke 还必须显式设置 `FA_ALLOW_DIRTY=1`。

Smoke 示例：

```bash
FA_SHAPES='64x64 65x63' \
FA_CAUSAL='0 1' \
FA_WARMUP=2 \
FA_ITERATIONS=5 \
FA_REPEATS=1 \
FA_ALLOW_DIRTY=1 \
projects/flash_attention/scripts/benchmark.sh
```

`64×64` 验证 Async fast path；`65×63` 验证 N tail、causal 和 `fallback-tiled`。Smoke 只验证工具链，不产生性能结论。

## ncu 与 SASS 协议

ncu 使用 validate 模式，只 profile 一次目标 launch，避免 runner 的 correctness、warmup 和 benchmark launch 混在一起：

```bash
projects/flash_attention/scripts/profile.sh tiled 512 128 0
projects/flash_attention/scripts/profile.sh tiled-async 512 128 0
projects/flash_attention/scripts/profile.sh tiled 1024 128 0
projects/flash_attention/scripts/profile.sh tiled-async 1024 128 0
```

完整 `.ncu-rep` 与文本摘要保存在本地 `results/profiles/`，不提交。ncu 可能锁定时钟，并为不同 counter replay Kernel；ncu 的 Duration、命令总耗时和 replay 时间都不是正常墙钟。

SASS 单独提取：

```bash
projects/flash_attention/scripts/extract_sass.sh tiled
projects/flash_attention/scripts/extract_sass.sh tiled-async
```

完整 SASS 保存在本地 `results/sass/`；小型静态计数和关键片段保存在 `results/evidence/`。静态指令条数不等于运行时执行次数。

## 发布规则

公开结论必须同时绑定：GPU、shape、dtype、causal、实现路径和测量方法。必须同时保留：

- Async 改善 shape；
- Async 回退 shape；
- fallback shape；
- 波动较大、无法下结论的 shape。

不发布单一“平均 Async 加速”，不使用 ncu Duration 替代墙钟，也不因某个 profiler 指标改善就删除负结果。
