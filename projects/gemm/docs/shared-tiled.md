# Shared Memory Tiled GEMM

## 为什么做这一版

Naive GEMM 中，每个 thread 都直接从 global memory 读取 A 的一行和 B 的一列。同一个 block 内，不同 thread 会重复使用很多相同元素，但源码没有显式保存这些数据。

这一版把 K 维切成长度为 32 的 tile。每轮先由整个 block 协作把一块 A 和一块 B 搬到 shared memory，再完成当前 tile 的内积累加。这样，同一份 global memory 数据可以在 block 内重复使用。

## 线程与数据映射

当前配置为：

```text
TILE_SIZE = 32
block      = 32 × 32 threads
```

一个 block 计算 C 的一个 `32×32` tile，一个 thread 负责其中一个输出元素：

```text
row = blockIdx.y × 32 + threadIdx.y
col = blockIdx.x × 32 + threadIdx.x
```

第 `i` 轮中：

```text
A[row, i×32 + threadIdx.x] → tile_a[threadIdx.y][threadIdx.x]
B[i×32 + threadIdx.y, col] → tile_b[threadIdx.y][threadIdx.x]
```

加载完成后，每个 thread 读取 shared memory 中 A 的一行和 B 的一列，累加 32 次乘加。

## 两次同步分别解决什么

每轮 tile 有两个 `__syncthreads()`：

1. **加载后同步**：保证所有 A/B 元素都写入 shared memory，才能开始计算。
2. **计算后同步**：保证所有 thread 都读完当前 tile，才能让下一轮覆盖 shared memory。

缺少第二次同步时，较快的 warp 可能已经写入下一轮数据，而其他 warp 仍在读取上一轮数据，形成 shared memory 的 WAR（write-after-read）竞态。

## 尾块处理

M、N、K 不要求是 32 的整数倍：

- A 或 B 的 global 坐标越界时，对应 shared memory 位置写 `0.0f`。
- C 写回前检查 `row < M && col < N`。
- K 的最后一个 tile 可以只有部分元素有效，其余位置补零后仍执行固定 32 次累加。

## 我踩到的两个问题

第一次验收时，完整的 `32×32×32` 也对拍失败。原因是 B 的 global 行使用了 `threadIdx.x`，但数据写进了 `tile_b[threadIdx.y][threadIdx.x]`，global 行和 shared 行没有对应起来。修正为由 `threadIdx.y` 决定 B 行后，单 tile 对拍通过。

随后 racecheck 在 `K=65` 时报告 shared memory hazard。单轮 `K=17` 没有问题，是因为不存在下一轮覆盖；多轮计算需要在内积结束后再同步一次。补上第二个 `__syncthreads()` 后，racecheck 不再报告竞态。

## 正确性与安全检查

完整正确性矩阵共 11 个 shape，覆盖：

- 极小矩阵和非方阵
- warp/tile 边界
- M、N、K 尾块
- 多轮 shared tile

验收结果：

```text
正确性矩阵                         11/11 PASS
compute-sanitizer memcheck        0 errors
compute-sanitizer racecheck       0 hazards
compute-sanitizer synccheck       0 errors
compute-sanitizer initcheck       0 errors
```

代表性多轮尾块用例：

```bash
./build/projects/gemm/gemm_runner \
  --kernel shared \
  --m 65 --n 127 --k 65 \
  --mode validate \
  --warmup 1
```

对应 sanitizer 命令只需替换工具名：

```bash
compute-sanitizer \
  --tool racecheck \
  --error-exitcode=99 \
  ./build/projects/gemm/gemm_runner \
  --kernel shared \
  --m 65 --n 127 --k 65 \
  --mode validate \
  --warmup 1
```

## 阶段性性能对比

在 A100 上使用相同输入和计时参数测试 `384³`：

| 版本 | 延迟 | GFLOPS |
| --- | ---: | ---: |
| Naive | 0.048558 ms | 2332.18 |
| Shared tiled | 0.044995 ms | 2516.89 |

阶段性加速比：

$$
\frac{0.048558}{0.044995}\approx1.079
$$

这组结果只用于确认优化方向，暂不作为最终性能数据。`384³` 一共只有 144 个 blocks，平均约 1.33 个 block/SM；GPU 频率、运行波动和 cuBLAS 基线也还没有纳入正式实验。

Shared 版本目前只比 Naive 快约 1.08 倍。可能原因包括 cache 对 Naive 的帮助、两次 block 同步的开销，以及 1024 threads/block 的资源约束。这里只记录假设，后续会用 Nsight Compute 检查 DRAM/L1/shared throughput、occupancy 和 warp stall，再判断真正瓶颈。
