# K/V `cp.async` 双缓冲实验

## 结论

在 NVIDIA A100 80GB PCIe（`sm_80`）、FP32、单 batch/单 head 的当前教学实现中，K/V 16B `cp.async` 双缓冲成功生成硬件异步搬运指令，并显著降低 long-scoreboard stall，但**没有在所有 shape 上获得稳定墙钟收益**。

Canonical 结果呈现明显的规模边界：non-causal `D=128` 下，`N=512` 延迟增加 18.2%，`N=768` 近似持平，`N=1024` 延迟降低 17.6%；`D=64` 在 `N=512/768` 明显回退，`N=1024` 仍仅近似持平。主要代价是双 stage Shared Memory 从 17,484 B/block 增至 33,912 B/block，同时寄存器从 31 增至 39 registers/thread，降低了 active warps，并增加 CTA wave 数。

该版本作为成功落地但收益依赖 shape 的优化实验保留；不将局部改善表述为跨 shape 稳定加速。

## 单变量改动

对照版本：[同步 K/V 搬运 Tiled](../kernels/tiled.cu)

实验版本：[K/V `cp.async` 双缓冲 Tiled](../kernels/tiled_async.cu)

两版保持以下内容一致：

- `BC=16`；
- 一个 block 负责一条 query；
- block size 为 128；
- FP32 Online Softmax 数学；
- 线程 0 串行计算 tile max/exp/sum；
- causal、尾块和全 mask 语义；
- 零 external workspace。

实验版本只改变 K/V 的 global-to-shared 数据流：

1. prologue 预取第一个 K/V tile；
2. steady state 将 next tile 预取到另一个 Shared Memory stage；
3. 等待 current stage 完成并建立 CTA 可见性；
4. 消费 current stage 后 release；
5. 交换 `current_stage/current_valid` 与 `next_stage/next_valid`；
6. `D % 4 != 0` 或 K/V 基址不满足 16B 对齐时回退到同步 `tiled`。

Shared K/V 基址显式 16B 对齐；`MAX_D=128` 使行跨度为 512 B，`BC=16` 使 stage 跨度为 8,192 B，均保持 16B 对齐。

## Correctness 与 Safety

固定测试覆盖：

- 最小输入；
- `D % 4 != 0` fallback；
- 16B async fast path；
- causal / non-causal；
- `N=17/33/37` tile 边界；
- `D=127/128` feature 尾部与上界；
- 全负 Scores、零 Q/K 和全 mask tile；
- `workspace_bytes=0`。

运行：

```bash
projects/flash_attention/scripts/test_tiled_async.sh
projects/flash_attention/scripts/sanitize.sh full
```

当前结果：13/13 Async correctness PASS；统一 full sanitizer 共执行 19 个命令，memcheck、racecheck、synccheck、initcheck 均为 0 errors/hazards。

## SASS 证据

构建产物中已确认 Async Kernel 包含：

```text
LDGSTS.E.BYPASS.128
ARRIVES.LDGSTSBAR.64
```

这证明 16B global-to-shared 异步路径已生成对应的 Ampere 机器指令，而不仅是源码层使用 Pipeline API。

## Canonical 墙钟结果

协议：CUDA Event，warmup 10，iterations 50；每个版本独立重复 3 轮，表中取 latency 中位数。完整 48 行展示表与 causal 结果见 [generated benchmark](../results/generated/a100-fp32.md)；三次重复 latency 的整体 min/max 和全部实现的 spread 保存在 [raw CSV](../results/raw/a100-fp32.csv)。

| Shape | 同步 `tiled` | `tiled-async` | 结果 |
| --- | ---: | ---: | ---: |
| `N=512,D=64` | 0.300175 ms | 0.466780 ms | 延迟增加 55.5% |
| `N=768,D=64` | 0.563692 ms | 0.735867 ms | 延迟增加 30.5% |
| `N=1024,D=64` | 1.487606 ms | 1.457316 ms | 延迟降低 2.0%，近似持平 |
| `N=512,D=128` | 0.555213 ms | 0.656384 ms | 延迟增加 18.2% |
| `N=768,D=128` | 1.040323 ms | 1.027502 ms | 延迟降低 1.2%，近似持平 |
| `N=1024,D=128` | 2.455716 ms | 2.022994 ms | 延迟降低 17.6% |

该结果不能简化为“Async 更快”或“Async 更慢”。它说明隐藏 global dependency 的收益与双缓冲资源代价在不同 CTA 数量和 feature 维度下会改变主导关系。

## 资源变化

| 资源 | 同步 `tiled` | `tiled-async` |
| --- | ---: | ---: |
| Registers/thread | 31 | 39 |
| Shared memory/block | 17,484 B | 33,912 B |
| Local memory | 0 | 0 |

Async 没有 spill，但 Shared Memory 近乎翻倍，寄存器增加 8 个/thread。

## ncu 对比

以下数据只比较 stall 与 active-warps 组成；ncu 下的 Kernel 时间不作为正常墙钟。Shape 均为 `D=128,causal=0`。

| Shape | 指标 | 同步 `tiled` | `tiled-async` |
| --- | --- | ---: | ---: |
| `N=512` | Waves/SM | 0.53 | 1.19 |
| `N=512` | Active warps | 29.60% | 20.40% |
| `N=512` | Long scoreboard | 8.56 | 0.06 |
| `N=512` | Short scoreboard | 0.96 | 1.32 |
| `N=512` | Barrier stall | 4.44 | 4.55 |
| `N=512` | Warp latency | 19.20 cycles | 10.33 cycles |
| `N=1024` | Waves/SM | 1.05 | 2.37 |
| `N=1024` | Active warps | 46.76% | 21.63% |
| `N=1024` | Long scoreboard | 9.51 | 0.04 |
| `N=1024` | Short scoreboard | 1.62 | 1.32 |
| `N=1024` | Barrier stall | 7.03 | 4.63 |
| `N=1024` | Warp latency | 24.36 cycles | 10.23 cycles |

`cp.async` 将 long-scoreboard 几乎降到零，并降低 warp instruction latency；代价是 active warps 明显下降。`N=512` 时资源与 wave 代价占主导，`N=1024,D=128` 时隐藏 global dependency 的收益足以覆盖该代价。

## 决策

停止继续修改 Async Kernel 的数学与 tile 宽度，保留当前版本。Canonical 结论同时保留：

1. `D=64` 在 `N=512/768` 明显回退，`N=1024` 仍仅近似持平；
2. non-causal `D=128` 从 `N=768` 近似持平跨到 `N=1024` 的 17.6% 延迟降低；
3. causal `D=128,N=768` spread 为 5.74%，明确标记 inconclusive；
4. `N=512,D=128` 的回退与 `N=1024,D=128` 的改善使用相同 ncu/SASS 协议解释。
