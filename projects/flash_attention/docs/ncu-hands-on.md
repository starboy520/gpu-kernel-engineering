# Nsight Compute 完整实操：用 FlashAttention 学习性能分析

## 0. 这份实验要完成什么

本手册不是 ncu 指标词典，而是一套可以逐步执行的实验。你会使用本项目中的同步 Tiled 与 `cp.async` Tiled Async，完成：

1. 确认环境、构建和实际执行路径；
2. 先用 CUDA Event 得到正常墙钟结论；
3. 第一次采集并保存 `.ncu-rep`；
4. 使用 CLI 和 GUI 查看同一份报告；
5. 按资源 → occupancy → scheduler → stall → memory 的顺序分析；
6. 比较 `512×128` 回退案例与 `1024×128` 改善案例；
7. 使用 SASS 验证 `cp.async` 是否生成硬件指令；
8. 独立写出一份不混淆 benchmark、ncu 和 SASS 的性能结论。

实验对象：

| Shape | Canonical 墙钟现象 | 学习目的 |
| --- | --- | --- |
| `N=512,D=128,causal=0` | Async 延迟增加 18.2% | 指标改善但整体回退 |
| `N=1024,D=128,causal=0` | Async 延迟降低 17.6% | 检查更长流水是否与收益出现一致 |

两版保持 FP32、`BC=16`、128 threads/block、一个 CTA/query 和 Online Softmax 数学相同。主要变量只有 K/V 搬运方式：同步加载与双 stage 16B `cp.async`。

## 1. 先记住证据顺序

```text
Correctness / 实际 path
        ↓
CUDA Event benchmark：到底快没快
        ↓
Nsight Compute：为什么出现这个结果
        ↓
SASS：源码是否生成目标机器指令
```

| 证据 | 回答的问题 | 不能回答的问题 |
| --- | --- | --- |
| Runner validation | 输出是否正确、实际走哪个 path | 性能高低 |
| CUDA Event | 正常运行时谁更快、快多少 | 为什么快 |
| ncu | 资源、occupancy、stall、吞吐如何变化 | 正常墙钟 |
| SASS | 是否生成 `LDGSTS` 等目标指令 | 是否动态重叠、是否更快 |

> **一句话记忆：Benchmark 判断结果，ncu 解释机制，SASS 验证编译。**

## 2. Step 1：进入仓库并检查工具

从仓库根目录开始：

```bash
cd /home/qichengjie/workspace/gpu-kernel-engineering

git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
nvcc --version
ncu --version
command -v ncu
command -v ncu-ui
```

当前机器预期：

```text
GPU：NVIDIA A100 80GB PCIe
Compute capability：8.0
Nsight Compute CLI：2026.2
ncu：/usr/local/cuda/bin/ncu
ncu-ui：/usr/local/cuda/bin/ncu-ui
```

检查清单：

- [ ] `ncu --version` 正常输出；
- [ ] GPU 是 A100，compute capability 是 8.0；
- [ ] 没有其他训练或 benchmark 占用同一 GPU；
- [ ] 后续四次 profile 使用同一个 runner；
- [ ] 实验过程中不修改或重新编译 Kernel。

### 常见问题：没有计数器权限

若看到 `ERR_NVGPUCTRPERM`，说明当前用户无权访问 GPU performance counters。这不是 Kernel bug。需要管理员按 NVIDIA 指南开放 counters，或在允许的环境中运行。不要用删除指标、改 Kernel 的方式绕过权限问题。

## 3. Step 2：Fresh build 和正确性门槛

在 VS Code 中：

1. 选择 CMake preset `release-sm80`；
2. 构建 target `flash_attention_runner`；
3. 在 Test Explorer 运行 `flash_attention_tiled_validate` 与 `flash_attention_tiled_async_validate`。

命令行只用于确认 runner 存在和记录哈希：

```bash
runner=build/projects/flash_attention/flash_attention_runner

test -x "$runner"
sha256sum "$runner"
```

记录 SHA256。后面四份 ncu report 必须来自这个二进制。

## 4. Step 3：验证正确性和实际 path

逐个运行，不要跳过输出：

```bash
runner=build/projects/flash_attention/flash_attention_runner

"$runner" --kernel tiled \
    --n 512 --d 128 --causal 0 \
    --input-pattern random --mode validate \
    --warmup 0 --iterations 1 --seed 1234

"$runner" --kernel tiled-async \
    --n 512 --d 128 --causal 0 \
    --input-pattern random --mode validate \
    --warmup 0 --iterations 1 --seed 1234
```

必须看到：

```text
kernel=tiled path=tiled ... status=PASS ... workspace_bytes=0
kernel=tiled-async path=fast-pipeline-16b ... status=PASS ... workspace_bytes=0
```

再将 `N` 改为 1024，重复两次。

检查清单：

- [ ] 四次都是 `status=PASS`；
- [ ] Tiled 的 `path=tiled`；
- [ ] Async 的 `path=fast-pipeline-16b`；
- [ ] `workspace_bytes=0`。

### Path 练习：主动触发 fallback

```bash
"$runner" --kernel tiled-async \
    --n 512 --d 127 --causal 0 \
    --input-pattern random --mode validate \
    --warmup 0 --iterations 1 --seed 1234
```

预期：

```text
kernel=tiled-async path=fallback-tiled ... status=PASS
```

回答：为什么 registry 名还是 `tiled-async`，但不能将这次结果当成 Async fast path？

## 5. Step 4：先读正常墙钟结果

不要先打开 ncu。先阅读已经固化的 canonical 结果：

- [A100 canonical benchmark](../results/generated/a100-fp32.md)
- [Raw CSV](../results/raw/a100-fp32.csv)

本实验只抄两组 non-causal `D=128`：

| Shape | Tiled | Async | 常规延迟变化 |
| --- | ---: | ---: | ---: |
| `512×128` | 0.555213 ms | 0.656384 ms | 增加 18.2% |
| `1024×128` | 2.455716 ms | 2.022994 ms | 降低 17.6% |

常规延迟变化：

$$
\Delta T=100\frac{T_{async}-T_{tiled}}{T_{tiled}}
$$

先写下假设：

- `512×128`：为什么目标 dependency 可能改善，但整体仍回退？
- `1024×128`：为什么更长的 steady state 可能开始受益？

> ncu 的 Duration、命令总耗时和 replay 时间都不能替代以上 CUDA Event 数字。

## 6. Step 5：第一次采集 ncu report

先准备输出目录和本实验统一使用的指标集合：

```bash
runner=build/projects/flash_attention/flash_attention_runner
profile_dir=projects/flash_attention/results/profiles
mkdir -p "$profile_dir"

metrics='launch__registers_per_thread,launch__shared_mem_per_block_static,launch__waves_per_multiprocessor,launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warp_latency_per_inst_issued.ratio,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld'
```

现在执行第一条完整 ncu 命令，采集 `512×128` Tiled：

```bash
ncu \
    --force-overwrite \
    --replay-mode kernel \
    --cache-control all \
    --clock-control base \
    --kernel-name-base demangled \
    --kernel-name 'regex:.*tiled_attention_kernel\(' \
    --launch-count 1 \
    --metrics "$metrics" \
    --export "$profile_dir/tiled-512x128-causal0.ncu-rep" \
    "$runner" \
    --kernel tiled \
    --n 512 --d 128 --causal 0 \
    --input-pattern random \
    --mode validate \
    --warmup 0 --iterations 1 \
    --seed 1234
```

参数逐项解释：

| 参数 | 作用 |
| --- | --- |
| `--force-overwrite` | 覆盖同名 report，避免交互确认 |
| `--replay-mode kernel` | 为不同 counters replay 同一次 Kernel launch |
| `--cache-control all` | 每个 replay pass 统一刷新可控 cache 状态 |
| `--clock-control base` | profile 时锁定 base clock，提高指标可比性 |
| `--kernel-name-base demangled` | Kernel filter 使用可读的 C++ 函数名 |
| `--kernel-name 'regex:...'` | 只匹配同步 Tiled Kernel |
| `--launch-count 1` | 只收集一个匹配 launch |
| `--metrics "$metrics"` | 只采本实验统一的资源、stall 和吞吐指标 |
| `--export` | 保存二进制 `.ncu-rep` |
| `--mode validate` | Runner 只发起一次目标实现，不混入 warmup/timed launches |

终端会显示：

```text
==PROF== Connected ...
==PROF== Profiling "...tiled_attention_kernel..." ... 10 passes
kernel=tiled path=tiled ... status=PASS
==PROF== Report: ...tiled-512x128-causal0.ncu-rep
```

逐项解释：

- `Profiling "tiled_attention_kernel"`：Kernel filter 命中了正确函数；
- `10 passes`：为收集不同 counters，ncu replay 了 Kernel；
- `status=PASS`：被 profile 的运行仍通过 correctness；
- `.ncu-rep`：完整报告，不是普通文本日志。

确认文件：

```bash
ls -lh projects/flash_attention/results/profiles/
```

应该至少出现：

```text
tiled-512x128-causal0.ncu-rep
```

`.ncu-rep` 被 Git 忽略，不会污染公开仓库。自动化脚本会额外保存 `.txt` 命令日志；本节原始命令只生成 `.ncu-rep`。

## 7. Step 6：用 CLI 阅读一份 report

设定路径：

```bash
report=projects/flash_attention/results/profiles/tiled-512x128-causal0.ncu-rep
```

### 7.1 看自定义 metrics 的 Details 页面

```bash
ncu --import "$report" --page details | less
```

退出 `less`：按 `q`。Step 5 的原始 ncu 命令使用显式 `--metrics`，因此 Details 页面预期只有：

```text
Section: Command line profiler metrics
```

它不会自动生成 `Launch Statistics`、`Occupancy`、`Scheduler Statistics` 等标准 section 页面。这里先核对表中包含 Registers、SMEM、Waves、Active/Eligible、stall 和 throughput 指标。标准 section 与分析规则在 Step 10 另外采集。

### 7.2 看全部 raw metrics

```bash
ncu --import "$report" --page raw | less
```

Raw 页面字段很多，第一次不要从头读到尾。后面使用摘要器只提取本项目需要的指标。

### 7.3 导出 raw CSV

```bash
ncu --import "$report" --page raw --csv \
    > /tmp/tiled-512x128-raw.csv

head -3 /tmp/tiled-512x128-raw.csv
```

CSV 有三行语义：表头、单位、数值。不要忽略单位行。

## 8. Step 7：采集第一组对照并自动摘要

使用完全相同的 metrics/cache/clock 设置，只替换 Kernel filter、runner kernel 和输出文件，采集 `512×128` Async：

```bash
ncu \
    --force-overwrite \
    --replay-mode kernel \
    --cache-control all \
    --clock-control base \
    --kernel-name-base demangled \
    --kernel-name 'regex:.*tiled_async_attention_kernel\(' \
    --launch-count 1 \
    --metrics "$metrics" \
    --export "$profile_dir/tiled-async-512x128-causal0.ncu-rep" \
    "$runner" \
    --kernel tiled-async \
    --n 512 --d 128 --causal 0 \
    --input-pattern random \
    --mode validate \
    --warmup 0 --iterations 1 \
    --seed 1234
```

先用原生 ncu 查看两份 Details：

```bash
ncu --import "$profile_dir/tiled-512x128-causal0.ncu-rep" \
    --page details

ncu --import "$profile_dir/tiled-async-512x128-causal0.ncu-rep" \
    --page details
```

为了减少手抄错误，可以再运行项目摘要器；它内部执行的是 `ncu --import REPORT --page raw --csv`，不是新的 profile：

```bash
projects/flash_attention/scripts/summarize_ncu.py \
    projects/flash_attention/results/profiles/tiled-512x128-causal0.ncu-rep \
    projects/flash_attention/results/profiles/tiled-async-512x128-causal0.ncu-rep
```

当前环境的关键预期值：

| Metric | Tiled | Async | 第一层解释 |
| --- | ---: | ---: | --- |
| Registers/thread | 31 | 39 | Async pipeline 增加状态 |
| Static SMEM | 17.484 KB | 33.912 KB | 双 stage 近乎翻倍 |
| Waves/SM | 0.53 | 1.19 | Async 需要更多 wave |
| SMEM limit blocks/SM | 9 | 4 | Shared Memory 是主要 residency 代价 |
| Active warps | 29.60% | 20.40% | Async resident warps 更少 |
| Eligible warps/cycle | 0.346 | 0.408 | 少量 active warp 更容易处于可发射状态 |
| Long scoreboard | 8.56 | 0.06 | global dependency 几乎消除 |
| Short scoreboard | 0.96 | 1.32 | dependency 部分转向 Shared/MIO |
| Barrier stall | 4.44 | 4.55 | 同步等待没有改善 |
| Warp latency | 19.20 cycles | 10.33 cycles | 单 warp 依赖链改善 |

### 8.1 必须先看资源

A100 report 中的 SM Shared Memory 容量为 167,936 B。ncu 的 residency 计算还包含每 block 1,024 B driver reserve，因此应使用 allocated Shared Memory：Tiled 18,560 B/block，Async 34,944 B/block，而不是直接用源码静态数组大小。

$$
\left\lfloor\frac{167936}{18560}\right\rfloor=9\text{ blocks/SM}
$$

$$
\left\lfloor\frac{167936}{34944}\right\rfloor=4\text{ blocks/SM}
$$

每 block 为 4 warps，因此只按 Shared Memory 估计的理论 occupancy 上限约为 56.25% 与 25%。

### 8.2 再看 active 与 eligible

- Active warps：驻留/活跃 warp 数量；
- Eligible warps：本周期真正满足发射条件的 warp；
- occupancy 高不保证 eligible 高；
- eligible 高也不自动保证整体快，还要看 wave、工作量和指令效率。

### 8.3 再看 stall

按固定顺序：

1. Long scoreboard 是否下降；
2. Short scoreboard 是否上升；
3. Barrier 是否下降；
4. Warp latency 是否下降；
5. 最后回到 benchmark。

> Stall ratio 的单位是 `inst`，不是百分比。不能写“Long scoreboard 从 8.56% 降到 0.06%”。

### 8.4 得出 `512×128` 结论

正确写法：

> SASS 证明 `cp.async` 已 lowering 为硬件异步指令，long scoreboard 的显著下降支持 global dependency 等待得到改善；同时，双 stage Shared Memory 将 occupancy limit 从 9 降到 4 blocks/SM，并把 waves/SM 从 0.53 增至 1.19。资源与 wave 变化和正常 CUDA Event 延迟增加 18.2% 的结果一致，支持“资源/调度代价抵消局部收益”的解释，但尚未构成隔离因果证明。

错误写法：

> Long scoreboard 降了，所以 Async 优化成功。

## 9. Step 8：采集 `1024×128` 改善案例

```bash
ncu \
    --force-overwrite \
    --replay-mode kernel \
    --cache-control all \
    --clock-control base \
    --kernel-name-base demangled \
    --kernel-name 'regex:.*tiled_attention_kernel\(' \
    --launch-count 1 \
    --metrics "$metrics" \
    --export "$profile_dir/tiled-1024x128-causal0.ncu-rep" \
    "$runner" \
    --kernel tiled \
    --n 1024 --d 128 --causal 0 \
    --input-pattern random \
    --mode validate \
    --warmup 0 --iterations 1 \
    --seed 1234

ncu \
    --force-overwrite \
    --replay-mode kernel \
    --cache-control all \
    --clock-control base \
    --kernel-name-base demangled \
    --kernel-name 'regex:.*tiled_async_attention_kernel\(' \
    --launch-count 1 \
    --metrics "$metrics" \
    --export "$profile_dir/tiled-async-1024x128-causal0.ncu-rep" \
    "$runner" \
    --kernel tiled-async \
    --n 1024 --d 128 --causal 0 \
    --input-pattern random \
    --mode validate \
    --warmup 0 --iterations 1 \
    --seed 1234
```

原生读取：

```bash
ncu --import "$profile_dir/tiled-1024x128-causal0.ncu-rep" \
    --page details

ncu --import "$profile_dir/tiled-async-1024x128-causal0.ncu-rep" \
    --page details
```

可选自动摘要：

```bash
projects/flash_attention/scripts/summarize_ncu.py \
    projects/flash_attention/results/profiles/tiled-1024x128-causal0.ncu-rep \
    projects/flash_attention/results/profiles/tiled-async-1024x128-causal0.ncu-rep
```

关键预期值：

| Metric | Tiled | Async |
| --- | ---: | ---: |
| Waves/SM | 1.05 | 2.37 |
| Active warps | 46.76% | 21.63% |
| Eligible warps/cycle | 0.579 | 0.426 |
| Long scoreboard | 9.51 | 0.04 |
| Short scoreboard | 1.62 | 1.32 |
| Barrier stall | 7.03 | 4.63 |
| Warp latency | 24.36 cycles | 10.23 cycles |
| SM throughput | 24.99% | 30.95% |

和 `512×128` 对比：

- 每条 query 的 K/V tile 数从 32 增至 64；
- pipeline steady state 更长；
- Long scoreboard 仍接近零；
- Barrier、Warp latency 下降；
- SM throughput 上升；
- 尽管 occupancy 仍降低，其他依赖和吞吐指标的改善与墙钟收益同时出现；
- 正常 CUDA Event 延迟降低 17.6%。

这是支持性证据，不是严格因果证明。若要隔离 Shared Memory 代价，还需要“相同双 buffer、但不用 Async”的额外对照。

## 10. Step 9：在 GUI 中看报告

本机有 `ncu-ui`。如果当前 Linux 会话具备图形显示：

```bash
ncu-ui projects/flash_attention/results/profiles/tiled-async-1024x128-causal0.ncu-rep
```

也可以：

1. 启动 `ncu-ui`；
2. 选择 **File → Open**；
3. 打开目标 `.ncu-rep`。

若当前通过无图形 SSH/VS Code Remote 连接：

1. 将 `.ncu-rep` 下载到本地电脑；
2. 在本地安装同版本或兼容版本 Nsight Compute；
3. 使用本地 GUI 打开 report。

### GUI 阅读顺序

使用 Step 5/7/8 原始 `ncu --metrics` 命令生成的精简 report，在 GUI 中主要显示 **Command line profiler metrics**。此时先练习搜索、排序和查看指标定义，不要寻找尚未采集的 Launch/Occupancy/Scheduler 标准 section。

完成下一步的 detailed/full report 采集后，再按该节末尾的“标准 section GUI 阅读顺序”操作。

## 11. Step 10：学习 section sets

查看当前 ncu 提供的 sets：

```bash
ncu --list-sets
ncu --list-sections
```

当前可见：

| Set | 特点 | 使用建议 |
| --- | --- | --- |
| `basic` | Launch、Occupancy、SpeedOfLight 等基础 section | 第一次了解 Kernel |
| `detailed` | 增加 Compute、Memory、SourceCounters 与 Roofline 等 | 深入一次代表 shape |
| `full` | 在 detailed 基础上加入 SchedulerStats、WarpStateStats 等 | 只在学习完整 GUI section 时使用 |

前面的原始命令使用精简的自定义 metrics 列表，便于公平比较四份报告。学习 Launch、Occupancy、Compute、Memory 和 SourceCounters 时，可以额外采一份 detailed report：

```bash
runner=build/projects/flash_attention/flash_attention_runner

ncu --force-overwrite \
    --set detailed \
    --replay-mode kernel \
    --cache-control all \
    --clock-control base \
    --kernel-name-base demangled \
    --kernel-name 'regex:.*tiled_async_attention_kernel\(' \
    --launch-count 1 \
    --import-source yes \
    --export /tmp/tiled-async-1024x128-detailed.ncu-rep \
    "$runner" --kernel tiled-async \
    --n 1024 --d 128 --causal 0 \
    --input-pattern random --mode validate \
    --warmup 0 --iterations 1 --seed 1234
```

注意：

- `detailed` 会采集更多 counters，replay 次数和耗时增加；
- 不要把这份 report 的 Duration 与正常 benchmark 对比；
- Tiled 与 Async 若要横向比较，必须使用完全相同的 set/metrics、cache 和 clock 设置。

若要在 GUI 中直接查看 **Scheduler Statistics** 与 **Warp State Statistics** 标准 section，需要改用 `--set full`。Full set 在当前 ncu 2026.2 预计采集约 7,381 个 metrics，成本很高；本教程的公平对照优先使用前面的精简指标，只有学习 GUI 页面时才额外采一份 full report。

### 标准 section GUI 阅读顺序

采集 detailed/full report 后再打开对应 report：

1. **Summary / GPU Speed Of Light**：先看整体 SM 和 Memory 吞吐；
2. **Launch Statistics**：看 grid、block、registers、static SMEM、waves；
3. **Occupancy**：看理论 occupancy 和 limiting factor；
4. **Compute Workload Analysis**：看计算管线利用；
5. **Memory Workload Analysis**：看 L1TEX/L2/DRAM 与 Shared Memory；
6. **Source**：查看源码、SASS 和 counter 关联；
7. **Scheduler Statistics（full）**：看 active/eligible/issued warps；
8. **Warp State Statistics（full）**：看 Long/Short Scoreboard 与 Barrier。

GUI 中点击指标名称可以查看定义说明。先确认定义和单位，再抄数值。

## 12. Step 11：查看 Shared Memory bank conflict

摘要器展示：

- `Shared load conflicts`；
- `Shared load wavefronts`；
- `Shared load wavefront factor`。

本手册使用诊断比值：

$$
\text{wavefront factor}
=
\frac{\text{wavefronts}}
{\text{wavefronts}-\text{bank conflicts}}
$$

只在相同 shape 和相同动态工作量下比较。不要直接比较 `N=512` 与 `N=1024` 的 conflict 总数，因为后者执行更多 tile。

这项比值用于观察 Shared load 的 wavefront 放大，不应当作所有 Shared Memory 指令的统一“冲突倍数”。Async copy 自身的 `LDGSTS` counters 还可以进一步使用 `shared_op_ldgsts` 指标研究。

## 13. Step 12：用 SASS 验证机器指令

先查看每个 Kernel 的静态资源：

```bash
cuobjdump --dump-resource-usage "$runner" 2>/dev/null | \
    grep -A2 -E 'tiled_attention_kernel|tiled_async_attention_kernel'
```

预期看到：

```text
Tiled：REG:31 SHARED:17484 LOCAL:0
Async：REG:39 SHARED:33912 LOCAL:0
```

注意这里是编译期 static Shared Memory；ncu occupancy 使用的 allocated per-block 还包含 driver reserve。

### 13.1 导出完整 SASS

```bash
mkdir -p /tmp/fa-sass
cuobjdump --dump-sass "$runner" > /tmp/fa-sass/all.sass
```

确认目标函数都存在：

```bash
grep 'Function :' /tmp/fa-sass/all.sass | \
    grep -E 'tiled_attention_kernel|tiled_async_attention_kernel'
```

### 13.2 截取 Tiled 函数

```bash
awk '
    /^[[:space:]]*Function[[:space:]]*:/ {
        if (found) exit
        if (index($0, "tiled_attention_kernel")) found=1
    }
    found { print }
' /tmp/fa-sass/all.sass > /tmp/fa-sass/tiled.sass
```

### 13.3 截取 Async 函数

```bash
awk '
    /^[[:space:]]*Function[[:space:]]*:/ {
        if (found) exit
        if (index($0, "tiled_async_attention_kernel")) found=1
    }
    found { print }
' /tmp/fa-sass/all.sass > /tmp/fa-sass/tiled-async.sass
```

`awk` 的逻辑：遇到包含目标名的 `Function :` 后开始打印，遇到下一个函数头时停止。

### 13.4 查找异步指令

```bash
grep -E 'LDGSTS|LDGSTSBAR' /tmp/fa-sass/tiled-async.sass
```

再确认同步版没有相同指令：

```bash
grep -E 'LDGSTS|LDGSTSBAR' /tmp/fa-sass/tiled.sass || \
    echo 'Tiled 中没有 LDGSTS/LDGSTSBAR'
```

### 13.5 统计静态 opcode

```bash
grep -E -c '[[:space:]]LDGSTS(\.[A-Z0-9]+)*[[:space:]]' \
    /tmp/fa-sass/tiled-async.sass

grep -E -c '[[:space:]]LDGSTS\.E\.BYPASS\.128[[:space:]]' \
    /tmp/fa-sass/tiled-async.sass

grep -E -c '[[:space:]]ARRIVES\.LDGSTSBAR(\.[A-Z0-9]+)*[[:space:]]' \
    /tmp/fa-sass/tiled-async.sass

grep -E -c '[[:space:]]LDL(\.[A-Z0-9]+)*[[:space:]]' \
    /tmp/fa-sass/tiled-async.sass || true

grep -E -c '[[:space:]]STL(\.[A-Z0-9]+)*[[:space:]]' \
    /tmp/fa-sass/tiled-async.sass || true
```

当前 Async 预期静态计数：`LDGSTS*=4`、`LDGSTS.E.BYPASS.128=4`、`ARRIVES.LDGSTSBAR*=2`、`LDL=0`、`STL=0`。

Async 预期包含：

```text
LDGSTS.E.BYPASS.128
ARRIVES.LDGSTSBAR.64
```

含义：

- `LDGSTS.E.BYPASS.128`：Ampere 128-bit global-to-shared 异步 copy；
- `ARRIVES.LDGSTSBAR.64`：异步 copy barrier 协议；
- `LDL/STL=0`：没有明显 local-memory spill 静态信号。

边界：

- 源码有 `cuda::memcpy_async()` 不保证生成硬件 Async；
- SASS 有 `LDGSTS` 证明 lowering；
- 静态 opcode 数不等于运行时执行次数；
- 有 `LDGSTS` 仍不能推出动态 overlap 或性能提高。

项目中的 `extract_sass.sh` 只是将以上命令自动化，并把小型证据写入 `results/evidence/`；学习阶段应先亲自执行本节原始 `cuobjdump`/`awk`/`grep` 命令。

## 14. 你的实验记录表

按顺序填写：先 benchmark，再 ncu，最后 SASS。

| Shape | Kernel | Path | Median ms | Spread | Regs | SMEM | Waves | Active | Eligible | Long | Short | Barrier | Warp latency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512×128` | Tiled | | | | | | | | | | | | |
| `512×128` | Async | | | | | | | | | | | | |
| `1024×128` | Tiled | | | | | | | | | | | | |
| `1024×128` | Async | | | | | | | | | | | | |

再写两个不超过四句话的结论：

### `512×128` 结论

```text
1. 正常墙钟：
2. Async 是否真实生成：
3. 改善的指标：
4. 为什么最终回退：
```

### `1024×128` 结论

```text
1. 正常墙钟：
2. Async 是否真实生成：
3. 改善与恶化的指标：
4. 为什么最终改善：
```

## 15. 常见错误排查

### ncu 没有命中 Kernel

检查输出是否出现 `Profiling "...目标函数..."`。重新核对 Step 5/7/8 原始命令中的 `--kernel-name-base demangled` 和目标 regex。

### report 已存在

原始命令包含 `--force-overwrite`，同 shape report 会被覆盖。需要保留历史版本时先复制或重命名。

### ncu 运行很慢

这是 counter replay 的正常现象。指标越多、set 越大，pass 越多。不要因此减少某一侧的指标集合，否则对照协议不一致。

### GUI 没有源码

普通 profile 主要采自定义 counters。需要 Source 页面时使用 `--set detailed --import-source yes` 重新采集，并确保构建保留 `-lineinfo`。

### 指标显示 `n/a`

可能原因：当前芯片不支持、指标未采集、指标自动展开为 `.sum/.avg`，或 Kernel 没有相应事件。先执行：

```bash
ncu --query-metrics --chip ga100 | grep '指标关键词'
```

### 结果和手册预期略有不同

允许小幅波动。首先核对 GPU、driver、ncu 版本、commit、runner SHA、shape、causal、seed、path 和 metrics 集合。不要只挑符合旧数据的一轮。

## 16. 最终自测题

1. 为什么 `D=127` 不能证明 Async fast path 的性能？
2. 为什么 ncu 使用 validate，而不是 runner benchmark 模式？
3. `passes=10` 是 Kernel 正常执行十次吗？
4. Active warps 与 Eligible warps有什么区别？
5. Stall ratio 为什么不能写成百分比？
6. `LDGSTS` 出现后，为什么墙钟仍可能回退？
7. 为什么不能跨 shape 比较原始 bank-conflict 总数？
8. `N=512` 中 Waves/SM 从 0.53 到 1.19 意味着什么？
9. 为什么 `N=1024` occupancy 更低却仍可能更快？
10. benchmark、ncu、SASS 三类证据各自的边界是什么？

## 17. 最终四问模板

每轮性能分析都回答：

1. **假设是什么？** 例如用 Async 隐藏 K/V global-load dependency。
2. **只改了什么？** K/V 搬运改为双 stage 16B pipeline。
3. **指标支持吗？** SASS、resources、waves、active/eligible、stall 如何变化。
4. **墙钟改善吗？** 明确列出改善、回退、近似持平和 inconclusive shape。

任何一问缺少证据，都不要发布“优化成功”。
