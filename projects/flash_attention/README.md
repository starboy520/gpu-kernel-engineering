# FP32 FlashAttention 数据流重建

**状态：开发中。当前已完成 Naive Materialized、Online Tiled、Warp 并行归约与 K/V `cp.async` 双缓冲实验。**

本项目从标准 Scaled Dot-Product Attention 出发，在 NVIDIA A100（`sm_80`）上逐步重建不物化完整 `N×N` Scores 的 FlashAttention 数据流。当前定位是 **educational/research FP32 forward implementation**，重点是独立手写、正确性边界和可解释的性能工程过程，不宣称生产级 FlashAttention-2。

## 当前实现边界

```text
GPU：NVIDIA A100 80GB PCIe，sm_80
输入/累加：FP32
布局：单 batch、单 head、row-major
Q/K/V：[N,D]
Output：[N,D]
D：1..128
模式：causal / non-causal
尺寸：支持 N 非 tile 整除
方向：forward only
```

暂不包含多 batch、多 head、变长序列、backward、dropout、FP16/BF16 Tensor Core、RoPE fusion、FlashAttention-2 映射和 Hopper TMA/WGMMA。

## 迭代路线与当前进度

| 里程碑 | 状态 | 核心目标 | 当前证据 |
| --- | :---: | --- | --- |
| 工程与 reference | ✅ | CPU double reference、统一 runner、validation | [common/runner tests](tests/) |
| Naive Materialized | ✅ | `QK^T → Softmax → PV`，显式 `N×N` workspace | [18 组 correctness](tests/correctness_cases.csv)、[完整 sanitizer](scripts/sanitize.sh) |
| Online Tiled | ✅ | K/V 按 `BC=16` 分块，维护 `m/l/O_acc`，不物化 `N×N` | [13 组 correctness](scripts/test_tiled.sh)、零 workspace、[完整 sanitizer](scripts/sanitize.sh) |
| 并行归约 | ✅ | warp shuffle 并行 max/exp/sum | [正确性、安全性与负性能结果](docs/parallel-reduction.md) |
| `cp.async` 双缓冲 | ✅ | 双 stage 16B K/V 异步预取 | [Correctness、安全性、SASS 与混合性能结果](docs/async-pipeline.md) |
| 统一 benchmark | ⬜ | Naive/Tiled/Pipelined 公平墙钟比较 | 计划中 |
| ncu 与 SASS | ⬜ | 分析 scoreboard、吞吐、资源与机器指令 | 计划中 |
| 作品化收口 | ⬜ | 方法文档、正式结果、复现脚本、限制说明 | 计划中 |

这里的状态以仓库中可以重新运行的代码和测试为准，不以计划或历史实验数据代替当前证据。

## 已完成：Naive Materialized Attention

Naive 版本分三个 Kernel：

```text
Q[N,D] × K^T[D,N] → Scores[N,N]
Scores 每行 Stable Softmax
Probabilities[N,N] × V[N,D] → Output[N,D]
```

该版本故意在 global memory 中保存 `N×N` Scores，作为后续版本的正确性、workspace 和墙钟基线。

- Kernel：[kernels/naive.cu](kernels/naive.cu)
- Workspace：`N × N × sizeof(float)`
- 支持 causal / non-causal
- 支持非整除尺寸
- Launcher 保持异步，由 runner 负责同步和计时

## 已完成：Online Tiled Attention

Tiled 版本采用一个 block 负责一条 query，K/V 按 `BC=16` 分块读取。每条 query 只维护：

```text
当前 K/V tile
当前 tile Scores
running max m
running denominator l
running output accumulator O_acc[D]
```

每个 tile 使用：

$$
m_{new}=\max(m_{old},m_{tile})
$$

$$
\alpha=e^{m_{old}-m_{new}}
$$

$$
l_{new}=\alpha l_{old}+\sum_j e^{s_j-m_{new}}
$$

$$
O_{acc,new}=\alpha O_{acc,old}+\sum_j e^{s_j-m_{new}}V_j
$$

这里的求和只覆盖当前 tile 中有效且未 mask 的 key。若 causal mask 使整个 tile 无效，则该 tile 作为空贡献处理：`alpha=1`、tile weights 全为 0，并保持 `m/l/O_acc` 不变，避免计算 $-\infty-(-\infty)$。

全部 key 处理完成后：

$$
O=O_{acc}/l
$$

- Kernel：[kernels/tiled.cu](kernels/tiled.cu)
- External workspace：`0` bytes
- 支持 causal 全 mask tile
- 支持最后一个不完整 K/V tile
- 当前 tile max/sum 由线程 0 串行计算，作为并行归约实验的对照基线

FlashAttention 没有把主导计算从 $O(N^2D)$ 降为 $O(ND)$；当前实现减少的是完整 `N×N` 中间矩阵的额外存储和 HBM 往返。

## 已完成：Warp 并行归约实验

并行版本保持 `BC=16`、K/V 布局、Online Softmax 数学和 `acc` 更新不变，只将 tile max/exp/sum 改为 warp 0 内的 shuffle 归约与状态广播。

- Kernel：[kernels/tiled_parallel.cu](kernels/tiled_parallel.cu)
- Correctness：[13 组固定回归](scripts/test_tiled_parallel.sh)
- External workspace：`0` bytes
- Safety：[统一 sanitizer](scripts/sanitize.sh)
- 结果：[Warp 并行 Online Softmax 归约实验](docs/parallel-reduction.md)

该实验没有获得稳定收益：`D=64` 仅改善约 1%，`D=128` 持平或回退约 4.4%；ncu 中 barrier、long/short scoreboard 和 warp latency 均未改善。原因是当前 tile 只有 16 个 Scores，串行工作本来较小，而 shuffle、状态广播、控制流和更高寄存器压力抵消了收益。该版本作为可复现负结果保留，不继续优化 `BC=16` reduction。

## 已完成：K/V `cp.async` 双缓冲实验

Async 版本保持 `BC=16`、一个 block/query 和串行 Online Softmax 不变，只将 K/V global-to-shared 搬运改为双 stage 16B Pipeline；不满足 `D % 4 == 0` 或 K/V 基址对齐条件时回退到同步 Tiled。

- Kernel：[kernels/tiled_async.cu](kernels/tiled_async.cu)
- Correctness：[13 组固定回归](scripts/test_tiled_async.sh)
- External workspace：`0` bytes
- Safety：[19-command full sanitizer](scripts/sanitize.sh)
- 机器指令：`LDGSTS.E.BYPASS.128`、`ARRIVES.LDGSTSBAR.64`
- 结果：[K/V `cp.async` 双缓冲实验](docs/async-pipeline.md)

探索性墙钟呈现 shape 依赖：`N=1024` 改善约 1.9%～17.7%，`N=512` 回退约 18.1%～55.8%。ncu 显示 long scoreboard 几乎降为零，但 Shared Memory 从 17,484 B/block 增至 33,912 B/block，active warps 明显下降。该版本证明异步路径有效，但不宣称稳定加速。

## 当前验证证据

### Correctness

Tiled、Tiled Parallel 与 Tiled Async 固定测试矩阵均覆盖：

- 最小输入：`N=1,D=1`
- 小规模可定位输入：`N=3,D=2`
- K/V tile 边界：`N=17`、`N=33`、`N=37`
- feature 尾部与上界：`D=127`、`D=128`
- causal / non-causal
- 全负 Scores
- 零 Q/K
- `workspace_bytes=0`

所有 GPU 输出都与同一输入下的 CPU double reference 对拍。

### Safety

当前 Naive、Tiled、Tiled Parallel 和 Tiled Async 均已运行：

- Compute Sanitizer memcheck
- racecheck
- synccheck
- initcheck

验证对象包含 causal 和非整除 `N=37,D=24`；Tiled Parallel 包含 `N=33,D=128` 的全 mask shuffle memcheck，Tiled Async 包含 tail non-causal 与全 mask fast-path memcheck。仓库 full sanitizer 共执行 19 个命令，统一覆盖当前四条路径。

## 构建与当前验证

从仓库根目录执行：

```bash
cmake --preset release-sm80
cmake --build --preset release-sm80 --target flash_attention_runner

projects/flash_attention/scripts/validate.sh
projects/flash_attention/scripts/test_tiled.sh
projects/flash_attention/scripts/test_tiled_parallel.sh
projects/flash_attention/scripts/test_tiled_async.sh
projects/flash_attention/scripts/sanitize.sh full
```

也可以通过 VS Code CMake Tools 选择 `release-sm80` preset，构建 `flash_attention_runner`，并在 Test Explorer 中运行 `flash_attention_tiled_validate`。

## 下一步

并行归约与 `cp.async` 双缓冲实验已经结束。下一阶段：

1. 固化统一 benchmark 协议与结果输出；
2. 扩充 canonical shape 矩阵，确认 Async 收益边界；
3. 汇总 Naive/Tiled/Parallel/Async 的正常墙钟；
4. 最后完成 ncu、SASS 与作品化说明。

`BC=32` 不作为当前主线；仅当后续证据表明 tile 宽度是必要变量时再单独评估。

## 项目原则

1. 每个核心 Kernel 从手写实现出发。
2. correctness 和 sanitizer 通过后才做性能结论。
3. ncu、SASS 和正常墙钟分别采集，不能互相替代。
4. 优化没有变快时保留负结果，解释瓶颈如何转移。
5. 所有公开结论必须绑定 GPU、shape、dtype、实现路径和测量方法。
