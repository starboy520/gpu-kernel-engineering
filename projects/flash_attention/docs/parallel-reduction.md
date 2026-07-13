# Warp 并行 Online Softmax 归约实验

## 结论

在 A100 80GB PCIe（`sm_80`）、FP32、单 batch/单 head 的当前教学实现中，将每个 `BC=16` tile 的线程 0 串行 max/exp/sum 改为 warp shuffle 并行归约，**没有获得稳定墙钟收益**：`D=64` 仅改善约 1%，`D=128` 持平或回退约 4.4%。

该版本作为负结果保留。它证明 correctness 和同步语义成立，也说明对仅 16 个 Scores 的小归约，shuffle、广播、控制流和更高寄存器压力可能抵消并行收益。

## 单变量改动

对照版本：[串行归约 Tiled](../kernels/tiled.cu)

实验版本：[Warp 并行归约 Tiled](../kernels/tiled_parallel.cu)

两版保持以下内容一致：

- `BC=16`；
- 一个 block 负责一条 query；
- block size 为 128；
- Q/K/V Shared Memory 布局；
- Score 点积、causal mask、`m/l/O_acc` 数学；
- `acc` 更新和输出写回；
- 零 external workspace。

实验版本只改变当前 tile 的 Online Softmax 归约：

1. warp 0 使用 `warpReduceMaxF()` 得到 tile max；
2. lane 0 计算 `m_new`、`alpha` 和是否存在未 mask Score；
3. `__shfl_sync()` 将状态广播给 warp 0；
4. 每个有效 lane 计算一个指数权重；
5. `warpReduceSumF()` 得到 tile 指数和；
6. 整个 block 消费权重前保留必要的 `__syncthreads()`。

全 mask tile 使用 `alpha=1`、weight=0，并保持 `m/l/O_acc` 不变。

## Correctness 与 Safety

固定测试覆盖：

- `N=1,D=1`；
- causal / non-causal；
- `N=17/33/37` tile 边界；
- `D=127/128` feature 尾部与上界；
- 全负 Scores；
- 零 Q/K；
- `workspace_bytes=0`。

运行：

```bash
projects/flash_attention/scripts/test_tiled_parallel.sh
projects/flash_attention/scripts/sanitize.sh full
```

当前结果：13/13 correctness PASS；memcheck、racecheck、synccheck、initcheck 均为 0 errors/hazards。

## 探索性墙钟结果

协议：CUDA Event，warmup 10，iterations 50；每个版本独立重复 3 轮，表中取 latency 中位数。以下数据用于判断优化方向，尚未纳入最终 canonical benchmark。

| Shape | 串行 `tiled` | `tiled-parallel` | 结果 |
| --- | ---: | ---: | ---: |
| `N=512,D=64` | 0.300114 ms | 0.297411 ms | 快约 0.9% |
| `N=1024,D=64` | 1.487483 ms | 1.475236 ms | 快约 0.8% |
| `N=512,D=128` | 0.556298 ms | 0.556196 ms | 基本持平 |
| `N=1024,D=128` | 2.459423 ms | 2.568110 ms | 慢约 4.4% |

复现单次测量：

```bash
runner=build/projects/flash_attention/flash_attention_runner

$runner --kernel tiled --n 1024 --d 64 --causal 0 \
  --input-pattern random --mode benchmark --warmup 10 --iterations 50

$runner --kernel tiled-parallel --n 1024 --d 64 --causal 0 \
  --input-pattern random --mode benchmark --warmup 10 --iterations 50
```

## ncu Stall 对比

Shape：`N=1024,D=64,causal=0`。ncu 时间不作为正常墙钟，仅比较 stall 组成。

| 指标 | 串行 `tiled` | `tiled-parallel` | 变化 |
| --- | ---: | ---: | ---: |
| Barrier stall | 7.99 | 8.23 | 上升 |
| Long scoreboard | 6.36 | 7.43 | 上升 |
| Short scoreboard | 1.44 | 1.98 | 上升 |
| Warp latency | 21.94 cycles | 24.28 cycles | 上升 |

```bash
for kernel in tiled tiled-parallel; do
  ncu --launch-skip 1 --launch-count 1 --metrics \
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warp_latency_per_inst_issued.ratio \
  build/projects/flash_attention/flash_attention_runner \
  --kernel "$kernel" --n 1024 --d 64 --causal 0 \
  --input-pattern random --mode benchmark --warmup 0 --iterations 1
done
```

## 资源变化

| 资源 | 串行 `tiled` | `tiled-parallel` |
| --- | ---: | ---: |
| Registers/thread | 31 | 35 |
| Shared memory/block | 17,484 B | 17,484 B |
| Local memory | 0 | 0 |

并行版没有增加 Shared Memory 或 spill，但每线程增加 4 个寄存器。

## 原因分析

当前 tile 只有 16 个 Scores。串行版本每个 tile 的 max/exp/sum 工作量本来就小，而并行版引入：

- warp max shuffle；
- 三次 lane 0 状态广播；
- warp sum shuffle；
- 额外控制流；
- 更高寄存器压力。

串行基线的 max/exp/sum 原本已由线程 0 连续执行，两者都只在整个 block 消费 `score[]` 和 `alpha` 前进行一次 block 同步，因此并行化没有减少 block barrier 数量。实测 barrier stall 反而略升，说明小归约的 shuffle/广播开销和其余 CTA 内负载不均衡没有改善这一等待。

## 决策

停止继续优化 `BC=16` 的 Softmax reduction，保留该版本作为负结果。下一阶段不修改 Online Softmax 数学和 tile 宽度，转而评估：

1. K/V `cp.async` 双缓冲；
2. SASS 中的异步搬运指令；
3. long scoreboard、Shared Memory conflict、occupancy 与正常墙钟的共同变化。

`BC=32` 暂不进入主线，避免在异步搬运实验中同时改变 tile 宽度。
