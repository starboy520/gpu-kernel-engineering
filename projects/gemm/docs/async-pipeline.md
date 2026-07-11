# `cp.async` 双缓冲 GEMM

## 目标

这一版保留 2D Register Tiling 的计算部分，只改变 global memory 到 shared memory 的搬运方式：

```text
Vectorized：global → float4 register → shared
Async：     global ──cp.async──▶ shared
```

shared memory 使用两个 stages。计算当前 tile 时，另一个 stage 预取下一块 A/B，尝试把 global memory 延迟隐藏在 outer-product 计算中。

## Pipeline 状态机

当前参数：

```text
BM=64, BN=64, BK=16
TM=8,  TN=4
STAGES=2
```

执行顺序：

```text
Prologue：
  预取 tile 0 → stage 0
  producer_commit

Steady state：
  预取下一 tile → next_stage
  producer_commit

  consumer_wait current_stage
  block.sync
  计算 current_stage
  block.sync
  consumer_release current_stage

  stage = next_stage

Epilogue：
  最后一轮不再提交下一 tile
  等待并计算最后一个 stage
```

`consumer_wait()` 管理 pipeline batch 是否完成；`block.sync()` 管理整个 block 对 shared memory 的可见性和覆盖时机。计算后的同步保证所有 thread 都读完当前 stage，下一轮才能安全复用该 buffer。

## Shared memory 布局

16B async copy 要求 destination 对齐。当前布局为：

```cpp
__shared__ __align__(16) float s_a[2][BM][BK];
__shared__ __align__(16) float s_b[2][BK][BN];
```

因为：

```text
BK=16 → A 每行 64 B
BN=64 → B 每行 256 B
```

每行起点都保持 16B 对齐。这里不能机械使用 `BK+1`、`BN+1` padding，否则行 stride 会变成 68 B 和 260 B，破坏后续行的 16B alignment。

当前布局可能仍有 shared bank conflict。A 的跨行映射和 B 的同一行 stride-4 读取需要后续 ncu 验证；不能为了 padding 破坏 async-copy 对齐。

## 4B Pipeline Checkpoint

第一版每个 task 只复制一个 `float`：

```cpp
cuda::memcpy_async(..., sizeof(float), pipe);
```

它用于先验证 pipeline 状态机，不追求最终性能。

验收结果：

```text
单 tile、双 tile、三 tile和 stage 复用均 PASS
memcheck / racecheck / synccheck / initcheck 全部通过
SASS 出现 LDGSTS 和 LDGSTSBAR
```

`2048³` 阶段数据：

| 版本 | 延迟 | GFLOPS |
| --- | ---: | ---: |
| Register tiled | 2.645443 ms | 6494.138184 |
| Async 4B | 1.928581 ms | 8908.035714 |
| Vectorized | 1.333514 ms | 12883.153789 |

4B pipeline 比 Register 快约 1.37×，但只达到 Vectorized 的约 69.1%。大量小粒度 async copy 和 pipeline 控制开销是候选原因，具体仍需 profiler 证明。

## 16B Async Copy

16B 版本把一个搬运任务改为 4 个连续 FP32。vector 映射保持作者实现：

```cpp
for (int i = tid; i < tileW * tileH / 4; i += n_thread) {
    int tile_row = i * 4 / tileW;
    int tile_col = i * 4 % tileW;
}
```

边界判断和补零都以 4 个元素为单位：

```text
整组 4 个元素有效 → async copy 16B
否则               → shared 连续 4 个位置写零
```

仅传 `sizeof(float4)` 时，CUDA 13.3 仍可能按照 `float*` 的 4B 类型对齐拆成多个 `LDGSTS.E`。最终代码使用：

```cpp
cuda::aligned_size_t<16>(sizeof(float4))
```

把已经由 launcher 和 shared layout 保证的 16B 对齐信息传给编译器。

最终 SASS：

```text
4 LDGSTS.E.BYPASS.128
2 LDGSTSBAR.64
```

`LDGSTS.E.BYPASS.128` 证明主 copy 生成了真正的 128-bit global-to-shared async 指令，而不是四条相邻的 4B copy。

## Fast Path 与 Fallback

16B fast path 要求：

```text
A/B 基地址满足 16B 对齐
K % 4 == 0
N % 4 == 0
BK % 4 == 0
BN % 4 == 0
```

M 不影响行 stride，可以不被 4 整除。

路径：

```text
满足条件   → kernel=async-16b, path=fast-pipeline-16b
不满足条件 → kernel=async-16b, path=fallback-register
```

## 正确性与安全检查

16B 版本覆盖：

```text
64×64×16    单 tile                       PASS
64×64×32    两个 stages                   PASS
64×64×48    stage 0 复用                  PASS
65×128×48   M 尾块 + stage 复用           PASS
65×132×36   M/N/K 尾块，仍满足 4 对齐     PASS
N=130       fallback-register             PASS
K=130       fallback-register             PASS
```

最终验收：

```text
CTest                         18/18 PASS
memcheck                      0 errors
racecheck                     0 hazards
synccheck                     0 errors
initcheck                     0 errors
fallback memcheck             0 errors
```

## 编译资源

```text
58 registers/thread
16424 bytes shared memory/block
0 spill stores
0 spill loads
128 threads/block
```

双缓冲 A/B 数据占：

$$
2\times(64\times16+16\times64)\times4=16384\text{ bytes}
$$

剩余空间来自 pipeline shared state，与 ptxas 报告相符。

## `2048³` 阶段结果

相同 `warmup=10`、`iterations=50`：

| 版本 | 延迟 | GFLOPS | 相对 Vectorized |
| --- | ---: | ---: | ---: |
| Async 4B | 1.928581 ms | 8908.035714 | 69.1% |
| Async 16B | 1.396777 ms | 12299.651688 | 95.5% |
| Vectorized | 1.333535 ms | 12882.956254 | 100% |

16B 相对 4B：

$$
\frac{1.928581}{1.396777}\approx1.381
$$

说明把 copy 粒度从 4B 提升到 16B 后，pipeline 快约 1.38×。

但 16B Async 仍比普通 Vectorized 慢约：

$$
\frac{1.396777}{1.333535}\approx1.047
$$

即约 4.7%。这次负结果很重要：异步流水的结构正确、SASS 也确认生成 128-bit async copy，但 pipeline 状态管理、双倍 shared memory、同步和当前 kernel 的计算/访存比例可能让重叠收益不足以抵消额外开销。具体原因要结合 ncu 的 long scoreboard、occupancy、registers、shared throughput 和 bank-conflict 指标判断。

## ncu：瓶颈从 global dependency 转到 shared memory

在相同 `2048³` shape 下，分别只 profile Async 16B 和 Vectorized 的目标 kernel：

| 指标 | Async 16B | Vectorized | 观察 |
| --- | ---: | ---: | --- |
| L1/TEX Throughput | 93.68% | 85.90% | Async 更接近 shared/L1 上限 |
| Compute Throughput | 65.98% | 68.37% | Async 的计算利用率略低 |
| Shared load conflict | 2.2-way | 1.3-way | Async 冲突更严重 |
| Shared load conflicts | 67.15M | 31.46M | Async 超过两倍 |
| Long scoreboard | 0.05 | 1.88 | Async 基本隐藏了 global-load 等待 |
| Short scoreboard | 1.87 | 0.49 | Async 等待 shared/MIO 更严重 |
| Achieved Occupancy | 38.49% | 26.79% | Async occupancy 反而更高 |
| Registers/thread | 58 | 83 | Async 寄存器更少 |

`cp.async` 的目标确实达到了：long-scoreboard stall 从 1.88 降到 0.05，下降约 97.3%。这说明 global-memory dependency 大部分被异步预取隐藏。

但性能没有超过 Vectorized，因为瓶颈发生了转移：Async 的 shared-load bank conflict 从 1.3-way 上升到 2.2-way，short-scoreboard stall 从 0.49 上升到 1.87，同时 L1/TEX throughput 被推到 93.68%。当前最强证据指向 shared-memory 访问，而不是 global-memory 等待或 occupancy 不足。

当前 A shared tile 没有 padding。一个 warp 横跨两个 `threadIdx.y`，两组 A 读取相差 8 行：

$$
8\times BK=8\times16=128\text{ floats}
$$

其 bank 偏移为：

$$
128\bmod32=0
$$

不同 shared 地址因此可能落到相同 bank。Vectorized 的 A tile 使用 `BK+1`，可以错开这类跨行映射。B 的冲突来自同一行内 `threadIdx.x×TN` 的 stride-4 读取，简单增加行 padding 不能解决。

因此当前结论是：

> 16B `cp.async` 双缓冲成功隐藏了 global-memory 延迟，但无 padding 的 shared layout 加重了 bank conflict，使瓶颈从 long scoreboard 转移到 short scoreboard。最终 Async 16B 比 Vectorized 慢约 4.7%。

## 为什么首版暂不做 swizzle

不能直接给 shared 数组加 `+1`：A/B 行 stride 会变成 68 B 和 260 B，后续行不再满足 16B async-copy destination alignment。要同时保留 16B 对齐并改变 bank 映射，需要设计以 vector 为单位的 shared-memory swizzle，并在写入和读取两端使用互相匹配的地址变换。

这会明显增加索引复杂度，也更接近 CUTLASS 的 shared-layout 设计。首版作品集的目标是展示完整的优化和分析方法，而不是强行让每一级都比上一版更快。当前版本已经具备完整证据链：

```text
正确性与 sanitizer 通过
          ↓
SASS 确认 128-bit LDGSTS
          ↓
ncu 确认 long scoreboard 下降
          ↓
同时发现 bank conflict 和 short scoreboard 上升
          ↓
解释墙钟性能为什么没有超过 Vectorized
```

因此 swizzle 作为后续扩展保留，不阻塞首版完成。若继续优化，应只改变 A 的 shared layout 做单变量实验，再用同一组 bank-conflict、short-scoreboard 和墙钟指标验证；不要同时修改 `TM/TN`、B layout 和 pipeline 状态机。

以上数据来自同一次阶段实验，不是最终简历数字。正式结果会重复多轮取中位数，并记录完整环境和 Git commit。
