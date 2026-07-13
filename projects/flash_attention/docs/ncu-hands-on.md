# Nsight Compute 实操：分析 FlashAttention `cp.async`

## 学习目标

完成本手册后，应能独立回答：

1. 正常 benchmark 与 ncu 各自回答什么；
2. 如何精确 profile 一个 Kernel，而不是混入 correctness/warmup launches；
3. occupancy、eligible warps、long/short scoreboard 和 barrier stall 分别意味着什么；
4. 为什么 `cp.async` 消除了 long scoreboard，墙钟仍可能回退；
5. 如何用 SASS 证明 `cuda::memcpy_async()` 生成了 Ampere 异步指令。

实验对象：

| Shape | 角色 |
| --- | --- |
| `N=512,D=128,causal=0` | Async 资源和尾波代价占主导的回退案例 |
| `N=1024,D=128,causal=0` | 流水更长、延迟隐藏可能覆盖资源代价的改善案例 |

两版都保持 FP32、`BC=16`、128 threads/block、一个 CTA/query 和同一 Online Softmax 数学；主要变量只有 K/V 同步搬运与双 stage 16B `cp.async`。

## 一、先建立证据顺序

```text
Correctness / 实际 path
        ↓
CUDA Event benchmark：快没快
        ↓
Nsight Compute：为什么
        ↓
SASS：源码是否落成机器指令
```

> **不要反过来。** 先看 profiler 再挑选支持预期的 benchmark，是结论偏差。

| 证据 | 能回答 | 不能回答 |
| --- | --- | --- |
| Runner validation | 输出是否正确、实际 path 是什么 | 性能 |
| CUDA Event | 正常运行延迟 | stall 原因 |
| ncu | 微架构指标如何变化 | 正常墙钟 |
| SASS | 是否生成 `LDGSTS` | 是否动态重叠、是否更快 |

## 二、冻结环境

从仓库根目录记录：

```bash
git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
nvcc --version
ncu --version
sha256sum build/projects/flash_attention/flash_attention_runner
```

检查项：

- GPU 是 A100，compute capability 为 8.0；
- 构建是 Release `sm_80`；
- 性能实验期间不修改源码、不重新构建；
- GPU 上没有训练或其他 benchmark；
- 所有对照使用同一 runner SHA256。

## 三、第一关：正确性和实际路径

```bash
runner=build/projects/flash_attention/flash_attention_runner

for n in 512 1024; do
    for kernel in tiled tiled-async; do
        "$runner" \
            --kernel "$kernel" \
            --n "$n" --d 128 --causal 0 \
            --input-pattern random \
            --mode validate \
            --warmup 0 --iterations 1 \
            --seed 1234
    done
done
```

必须确认：

- 四次都是 `status=PASS`；
- Tiled 是 `path=tiled`；
- Async 是 `path=fast-pipeline-16b`；
- 两者 `workspace_bytes=0`。

### Fallback 练习

```bash
"$runner" \
    --kernel tiled-async \
    --n 512 --d 127 --causal 0 \
    --input-pattern random \
    --mode validate \
    --warmup 0 --iterations 1
```

预期 `path=fallback-tiled`。此时 registry 名仍是 `kernel=tiled-async`，但实际执行的不是 Async fast path。因此性能记录必须同时保存 `kernel` 和 `path`。

## 四、第二关：先测正常墙钟

正式矩阵由脚本完成：

```bash
projects/flash_attention/scripts/benchmark.sh
```

学习时可以只运行两个 shape，但这会进入 smoke 输出：

```bash
FA_SHAPES='512x128 1024x128' \
FA_CAUSAL='0' \
FA_WARMUP=10 \
FA_ITERATIONS=50 \
FA_REPEATS=3 \
projects/flash_attention/scripts/benchmark.sh
```

记录 `tiled` 与 `tiled-async` 的中位数、min/max/spread。计算：

$$
S_{async}=\frac{T_{tiled}}{T_{async}}
$$

$$
\Delta_{async}=100(S_{async}-1)
$$

不要记录 ncu Duration。ncu 可能锁频并 replay Kernel，它不是用户正常运行时的延迟。

## 五、第三关：运行 ncu

### 5.1 一键采集

```bash
projects/flash_attention/scripts/profile.sh tiled 512 128 0
projects/flash_attention/scripts/profile.sh tiled-async 512 128 0
projects/flash_attention/scripts/profile.sh tiled 1024 128 0
projects/flash_attention/scripts/profile.sh tiled-async 1024 128 0
```

脚本使用：

- validate 模式：只有一次目标 Kernel launch；
- `--kernel-name`：精确过滤 Tiled 或 Async 函数；
- `--launch-count 1`：只采一个 launch；
- `--replay-mode kernel`：允许 counter replay；
- `--cache-control all` 和 `--clock-control base`：统一 profile 条件。

输出：

- `.ncu-rep`：可用 Nsight Compute GUI 打开；
- `.txt`：命令和 CLI 指标摘要。

### 5.2 为什么不用 benchmark 模式 profile

Runner benchmark 会依次执行：

1. correctness launch；
2. warmup launches；
3. timed launches。

依赖 `--launch-skip` 很脆弱：warmup 数改变就可能 profile 错 launch。validate 模式只有一次目标 launch，实验语义最清晰。

## 六、按固定顺序读指标

### 6.1 先看资源和 occupancy 限制

| 指标 | 含义 |
| --- | --- |
| `launch__registers_per_thread` | 每线程寄存器 |
| `launch__shared_mem_per_block_static` | 每 block 静态 Shared Memory |
| `launch__waves_per_multiprocessor` | CTA 相对 SM 容量形成的 wave 数 |
| `launch__occupancy_limit_shared_mem` | Shared Memory 限制的 blocks/SM |
| `launch__occupancy_limit_registers` | 寄存器限制的 blocks/SM |

当前静态资源预期：

| 资源 | Tiled | Async |
| --- | ---: | ---: |
| Registers/thread | 31 | 39 |
| Shared Memory/block | 17,484 B | 33,912 B |
| Local Memory | 0 | 0 |

A100 每 SM 最多约 163,840 B Shared Memory。只按 Shared Memory 粗估：

$$
\left\lfloor\frac{163840}{17484}\right\rfloor=9\text{ blocks/SM}
$$

$$
\left\lfloor\frac{163840}{33912}\right\rfloor=4\text{ blocks/SM}
$$

每 block 有 4 warps，Shared Memory 对理论 occupancy 的上限约为 56.25% 与 25%。实际值还受寄存器、block 数量和尾波影响。

### 6.2 再看 active 与 eligible warps

| 指标 | 含义 |
| --- | --- |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | resident/active warp 比例 |
| `smsp__warps_eligible.avg.per_cycle_active` | 每周期真正可发射的 warp 数 |

Active 高不保证快。大量 active warps 都在等待依赖时，eligible 仍可能很低。反之，Async occupancy 较低，但单个 warp 等待变少，也可能最终更快。

### 6.3 再看 stall

按以下顺序：

1. `long_scoreboard`：长延迟 global/local dependency；
2. `short_scoreboard`：常见于 Shared Memory/MIO dependency；
3. `barrier`：同步等待；
4. `warp_latency`：每条已发射指令对应的平均 warp latency。

指标全名：

```text
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warp_latency_per_inst_issued.ratio
```

> Stall ratio 不是百分比。不要把 `8.54` 写成 `8.54%`。

判断模板：

- Long 明显下降：`cp.async` 达成隐藏 global dependency 的局部目标；
- Short 上升：瓶颈可能转向 Shared Memory/MIO；
- Barrier 不降：流水没有减少该同步等待；
- Warp latency 下降：单 warp 依赖链改善；
- 最后仍必须回到 CUDA Event 判断墙钟。

### 6.4 最后看吞吐和 Shared Memory

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
lts__throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput.avg.pct_of_peak_sustained_elapsed
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld
```

原始 bank-conflict 数只能在相同 shape、相同动态工作量下比较。不能把 `N=512` 总 conflict 和 `N=1024` 总 conflict 直接比较。

## 七、用两个 shape 练习解释

### `N=512,D=128`

A100 有 108 个 SM：

$$
512/108\approx4.74\text{ CTAs/SM}
$$

Async 受 Shared Memory 限制约为 4 blocks/SM，可能产生额外尾波。即使 long scoreboard 接近零，双倍 Shared Memory、更低 active warps、pipeline 控制和尾波仍可能使墙钟回退。

回答模板：

> SASS 和 long scoreboard 证明 `cp.async` 已生效；但双 stage Shared Memory 降低 residency，并在 512 CTA 下形成不利波次。局部 dependency 改善没有覆盖资源与调度代价，因此正常 benchmark 回退。

### `N=1024,D=128`

$$
1024/108\approx9.48\text{ CTAs/SM}
$$

每条 query 处理 64 个 K/V tiles，steady state 更长。若 long scoreboard、barrier 和 warp latency 均改善，隐藏延迟的收益可能覆盖 occupancy 代价，从而使墙钟改善。

这是一组支持性证据，不是严格因果证明。若要进一步隔离 Shared Memory 代价，需要额外设计“相同双 buffer、但不用 Async”的对照。

## 八、第四关：查看 SASS

```bash
projects/flash_attention/scripts/extract_sass.sh tiled
projects/flash_attention/scripts/extract_sass.sh tiled-async
```

Async 预期包含：

```text
LDGSTS.E.BYPASS.128
ARRIVES.LDGSTSBAR.64
```

解释：

- `LDGSTS.E.BYPASS.128`：Ampere 128-bit global-to-shared async copy；
- `ARRIVES.LDGSTSBAR.64`：异步 copy barrier 协议；
- `LDL/STL=0`：没有明显 local-memory spill 静态信号。

源码出现 `cuda::memcpy_async()` 不代表编译器一定生成硬件 Async；SASS 出现 `LDGSTS` 证明 lowering，但仍不能证明动态 overlap 或性能提高。

## 九、填写自己的证据表

| Shape | Kernel | Path | Median ms | Spread | Regs | SMEM | Waves | Active | Eligible | Long | Short | Barrier | Warp latency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512×128` | Tiled | | | | | | | | | | | | |
| `512×128` | Async | | | | | | | | | | | | |
| `1024×128` | Tiled | | | | | | | | | | | | |
| `1024×128` | Async | | | | | | | | | | | | |

先填 benchmark，再填 ncu，最后填 SASS。

## 十、练习题

1. 为什么 `D=127` 不能用于证明 Async fast path 性能？
2. `LDGSTS` 出现后，为什么仍可能回退？
3. Active warps 下降是否必然变慢？
4. 为什么 stall ratio 不能写成百分比？
5. 为什么 ncu 使用 validate 而不是 benchmark 模式？
6. 为什么不能比较不同 shape 的原始 bank-conflict 总数？
7. 如何解释“long scoreboard 几乎归零，但 `N=512` 更慢”？
8. 如果 `N=768` 三次 spread 超过 3%，应该如何发布结论？

## 十一、最终四问总结

每次 profile 后固定回答：

1. **假设是什么？** 用 Async 隐藏 K/V global-load dependency。
2. **只改了什么？** K/V 搬运改为双 stage 16B pipeline。
3. **指标支持吗？** SASS 是否有 `LDGSTS`；long/short/barrier、occupancy 如何变化。
4. **墙钟改善吗？** 明确列出改善、回退与无法下结论的 shape。

如果四问中任何一问缺证据，就不要发布“优化成功”。
