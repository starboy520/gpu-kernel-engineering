# M1 Query-tiled vs M2 Warp-per-query 开发期对比

> 当前工作树包含开发中改动；本页用于决定下一轮优化，不作为 clean-commit canonical 发布结果。

## 协议

```text
GPU：NVIDIA A100 80GB PCIe，sm_80
实现：M1 query-tiled vs M2 warp-per-query
数据：FP32，batch=1，head=1，seed=1234
Shape：N=128/256/512/1024/2048，D=64/128，causal=0/1
计时：CUDA Event，warmup=10，iterations=50，repeats=3 取中位数
正确性：两条路径均先与 CPU double reference 对拍
```

完整开发期表位于 `results/generated/m2-development.md`。

## Benchmark 结论

### Non-causal

| Shape 范围 | 结果 |
| --- | --- |
| D=64 | 小 N 回退；1024 近似持平；2048 提升 9.7% |
| D=128 | 全部获益；256～2048 提升约 17.4%～27.8% |

M2 更适合 `D=128`：每个 Lane 有四个 feature slot，Warp QK reduction 的并行度更充分。

### Causal

Causal 路径多数回退：

- `D=64` 全部回退，约 15.9%～32.2%；
- `D=128` 在 N=128～1024 回退或近似持平；
- `2048x128 causal=1` 提升约 7.8%。

当前 causal 实现仍遍历所有未来 K/V tile，并对不同 Query Warp 形成不同的有效 Key 区域。Warp shuffle 与逐 Key 控制流成本在短序列上抵消了并行收益。后续优先研究 causal 整 Tile 跳过或减少全 mask 工作，而不是先改 non-causal 主路径。

## Nsight Compute

### 1024x128 non-causal

| Metric | M1 | M2 | 变化 |
| --- | ---: | ---: | ---: |
| ncu duration | 1.603 ms | 1.420 ms | M2 -11.4% |
| Registers/thread | 34 | 40 | +17.6% |
| Static SMEM/block | 20.784 KB | 16.384 KB | -21.2% |
| Achieved occupancy | 14.82% | 14.81% | 近似不变 |
| Eligible warps/cycle | 0.152 | 0.236 | +54.7% |
| Long scoreboard | 7.881 | 6.529 | -17.2% |
| Short scoreboard | 2.432 | 1.388 | -42.9% |
| Barrier stall | 2.566 | 0.090 | -96.5% |
| Warp latency | 17.353 cyc | 12.020 cyc | -30.7% |
| SM throughput | 13.63% | 19.68% | +44.4% |
| Shared bank conflicts | 29,974,528 | 0 | 消除 |

解释：`O_acc` 和 Query fragment 转入寄存器、Softmax 改成 Warp collective 后，Shared Memory 与 barrier 压力显著下降。虽然 registers/thread 从 34 增至 40，但当前 achieved occupancy 基本不变；eligible warps 和 SM throughput 上升，因此 non-causal `D=128` 获益。

### 1024x128 causal

M2 仍显著降低 long scoreboard、barrier 和 warp latency，并消除 bank conflict；但 ncu duration 仅从 1.322 ms 变为 1.338 ms，接近持平。这说明 causal 下动态工作量/控制流成本成为下一瓶颈，单看 stall 改善不能推出墙钟一定更快。

> ncu duration 只用于 profiler 内部比较；正式延迟以 CUDA Event 表为准。

## SASS

| Opcode | M1 | M2 |
| --- | ---: | ---: |
| `FFMA` | 140 | 102 |
| `SHFL` | 0 | 78 |
| `BAR` | 6 | 2 |
| `LDS` | 98 | 52 |
| `STS` | 72 | 2 |
| `LDL` | 0 | 0 |
| `STL` | 0 | 0 |
| `HMMA` | 0 | 0 |
| `LDGSTS` | 0 | 0 |

证据符合 M2 设计：

- 出现大量 `SHFL`，证明 Warp reduction/broadcast 路径存在；
- `BAR` 从 6 降到 2；
- Shared load/store 静态数量明显下降；
- 无 `LDL/STL`，当前寄存器 fragment 没有观察到 local spill；
- 仍是 FP32 SIMT，不含 Tensor Core 或 `cp.async`。

静态 opcode 数量不代表 runtime 执行次数。

## 当前判断

M2 的核心优化机制已经得到硬件证据支持：

```text
Shared O_acc → register O_acc
row leader Softmax → Warp collective
更多 SHFL → 更少 barrier/Shared traffic
```

下一轮只做一个变量：优化 causal 路径的未来 Tile/Key 工作量。Non-causal 主路径先保持不变。
