# Attention Lab 里程碑路线图

## 目标与边界

本路线图用于“一边学习、一边实现、一边形成性能证据”，不绑定周数。每个里程碑只有在完成晋级门槛后才进入下一阶段。

当前不会创建统一 `Attention Lab` 目录，也不会迁移已经完成的项目。未来开始具体里程碑时，再根据当时的代码边界决定目录结构。当前 [FP32 FlashAttention 基线](projects/flash_attention/) 保持冻结，作为后续所有实验的 correctness、性能方法和负结果基线。

路线图分为两条主线：

```text
Attention Lab
├── 已完成：Prefill Baseline
├── 主线 A：Advanced Prefill
│   ├── M1  Query-tiled FP32 SIMT
│   ├── M2  Warp-per-query
│   ├── M3  FP16/BF16 SIMT
│   ├── M4  MMA Playground
│   ├── M5  Tensor Core QK
│   ├── M6  Tensor Core PV
│   ├── M7  Shared Memory Swizzle
│   ├── M8  Multi-stage Pipeline
│   ├── M9  FlashAttention-2-style Mapping
│   ├── M10 Batch / Head / GQA / Varlen
│   └── M11 Backward
└── 主线 B：Inference Attention
    ├── D1 Decode Baseline
    ├── D2 KV Cache + Online Softmax
    ├── D3 Warp-level Decode
    ├── D4 MQA / GQA
    ├── D5 Split-KV
    ├── D6 PagedAttention
    └── D7 Continuous Batching
```

主线执行顺序为：先完成 Advanced Prefill，再进入独立 Decode/PagedAttention。Decode 是独立性能问题，不把它强行塞入 Prefill Kernel。

## 硬件契约

当前路线的主目标硬件是 NVIDIA A100（`sm_80`）。`m16n8k16` MMA、`ldmatrix`、`mma.sync` 和 `cp.async` 阶段均按 Ampere 能力设计；不在本路线中混入 Hopper TMA/WGMMA。

每个新 tile、dtype、stage 数或线程映射在实现前先计算 launch feasibility，并在构建后核对：

- registers/thread 与是否 spill；
- static/dynamic Shared Memory/block；
- block/SM 与 theoretical occupancy；
- grid 产生的 waves/SM；
- 目标 `sm_80` 指令是否实际生成。

## 学习所有权

每个里程碑继续遵循以下分工：

### 学习者亲手完成

- 核心 CUDA Kernel；
- 线程映射、索引和 Shared Memory 布局；
- Online Softmax、MMA fragment 或 Split-KV 状态的数据流；
- 关键同步和 pipeline stage 生命周期；
- 根据证据提出下一轮单变量优化假设；

### Assistant 可以完成

- CMake、registry、runner 与命令行接口；
- CPU/vendor reference；
- correctness cases、CTest 和 sanitizer 脚本；
- benchmark、ncu、PTX/SASS 取证工具；
- 失败复现、代码 review 与分层提示；
- 结果文档、对照表与提交收口。

### Review 提示层级

1. 指出需要检查的不变量或区域；
2. 指出错误的数据流、索引、同步阶段或内存假设；
3. 给出局部公式或伪代码，不直接替写完整 Kernel。

## 统一晋级门槛

每个实现型里程碑都必须经过：

```text
画出数学和数据流
→ 说明线程与数据所有权
→ 先准备失败的 correctness 测试
→ CPU/vendor reference 对拍
→ 边界 shape 与特殊输入通过
→ memcheck / racecheck / synccheck / initcheck
→ 固定协议 CUDA Event benchmark
→ ncu 验证瓶颈假设
→ PTX/SASS 验证关键指令
→ 保留收益、回退和无结论结果
→ 完成阶段工程总结
```

MMA Playground 等微实验可不运行完整 Attention benchmark，但必须有 fragment 级 correctness、资源与目标指令证据。

### 性能发布规则

任何性能结论都绑定：

- GPU 和 compute capability；
- dtype；
- batch、head、sequence length 和 head dimension；
- causal/non-causal；
- 实际实现路径；
- warmup、iterations、repeats 和统计量；
- Git commit 与 runner hash。

正常墙钟、ncu 和 SASS 分开采集。ncu Duration 不作为正常延迟；SASS 静态指令数不代表运行时执行次数。

## 已完成基线：Prefill Baseline

当前 [FP32 FlashAttention 数据流重建](projects/flash_attention/) 已完成：

- Naive Materialized Attention；
- 单 query/block Online Tiled Attention；
- Warp 并行归约负结果；
- K/V 16B `cp.async` 双缓冲混合结果；
- CPU double reference、CTest 和 full sanitizer；
- 48 行 A100 canonical benchmark；
- ncu 与 SASS 分层证据。

当前结构性限制是 `Br=1`：每个 CTA 只负责一条 Query，K/V 不能在多条 Query 间复用。这是 Advanced Prefill 的起点。

---

# 主线 A：Advanced Prefill

## M1：`Br=4` Query-tiled FP32 SIMT

### 为什么做

把当前 `Br=1,Bc=16` 的一维 K/V tiling 扩展为真正的二维 `Br×Bc` Attention tiling，使一个 CTA 处理多条 Query，并让一个 K/V tile 被多条 Query 复用。

### 保持不变

- FP32 输入与累加；
- `Bc=16`；
- SIMT 标量 FMA；
- Online Softmax 数学；
- 暂时不用 Tensor Core 和 `cp.async`；
- 单 batch、单 head、forward only。

### 学习者亲手实现

- `Q[Br,D]`、`K/V[Bc,D]` 和 `Score[Br,Bc]` 布局；
- 每条 Query 独立的 `m[row]`、`l[row]` 和 `O_acc[row,D]`；
- Query/K/V tail；
- causal mask 在二维 tile 中的语义；
- CTA 内线程到 Query/Key/Feature 的映射。

### 重点不变量

- 每条 Query 的 Online Softmax 状态绝不能串行污染其他 Query；
- K/V stage 在所有 Query 消费结束前不能覆盖；
- causal 全 mask tile 对每一行独立执行空贡献语义；
- `N % Br != 0` 和 `N % Bc != 0` 可同时出现。

### 证据门槛

- 对拍 `Br` 边界：`N=1/3/4/5/15/16/17/31/33`；
- 对拍 feature 边界：`D=1/2/63/64/65/127/128`；
- 至少覆盖 `N` tail 与 `D` tail 同时出现的组合；
- causal、non-causal、全负 Scores、零 Q/K；
- full sanitizer；
- 比较 K/V global load、Shared Memory、registers 和正常墙钟；
- ncu 观察 L1/L2、long scoreboard、barrier 和 occupancy。

### 晋级条件

能画出 `Br×Bc` 数据流，并证明 K/V 复用不是通过重复加载伪造；correctness、sanitizer 和最小 canonical 对照全部完成。

## M2：Warp-per-query 映射

### 为什么做

为每条 Query 建立明确的 Warp ownership，减少跨 Warp 状态共享，学习 Warp 内寄存器状态、shuffle 和线程职责设计。

### 初始设计

```text
CTA：4 warps
warp 0 → query row 0
warp 1 → query row 1
warp 2 → query row 2
warp 3 → query row 3
```

### 学习者亲手实现

- 一个 Warp 负责一条 Query；
- Warp 内 Score reduction 和 Online Softmax；
- `D=64/128` 时 feature 分片；
- 每 lane 的 `O_acc` 寄存器所有权；
- Warp 与 CTA barrier 的最小化。

### 证据门槛

- 完整复用 M1 的 `N`、`D`、双 tail 和特殊输入测试矩阵；
- synccheck 和 racecheck 重点覆盖 Warp/CTA 边界；
- 对比 block barriers、active/eligible warps、registers/thread；
- 解释 occupancy 下降是否由寄存器还是 Shared Memory 主导。

### 晋级条件

能够逐 lane 说明 Score、Softmax 状态和 Output fragment 的地址空间与生命周期，并用 ncu 解释同步变化。

## M3：FP16/BF16 SIMT，FP32 累加

### 为什么做

建立低精度数据与 FP32 数值状态的正确性基线，为 Tensor Core 版本提供可比较的 dtype 基线。

### 数据类型契约

```text
Q/K/V：FP16 或 BF16
Score dot accumulation：FP32
m/l：FP32
O_acc：FP32
Output：FP16/BF16（另保留 FP32 输出调试路径）
```

### 学习者亲手实现

- `half2` / `nv_bfloat162` 向量化加载与 unpack/convert；
- 低精度输入转换为 FP32 后使用 FP32 dot accumulation，不使用低精度累加替代该契约；
- 128-bit vectorized fast path 与安全 fallback；
- FP32 accumulator；
- 输出转换和 rounding；
- dtype dispatch。

### 证据门槛

- FP16 与 BF16 分开设容差；
- 极端 logits、长序列、全 mask、tail；
- 与 FP32 baseline 及 vendor reference 对比；
- SASS 验证低精度 load/convert 与 FP32 `FFMA` accumulation 路径；
- 记录误差随 `N` 的变化。

### 晋级条件

FP16/BF16 均有独立 correctness 与性能结果，能够解释误差来源和 fallback 条件。

## M4：MMA Playground

### 为什么做

在完整 Attention 之外单独掌握 Tensor Core fragment、`ldmatrix` 和 `mma.sync`，避免同时调试 Attention 数学与 MMA 布局。

### 微实验顺序

1. `m16n8k16` 单 MMA；
2. 多 K-step 累加；
3. Shared Memory → `ldmatrix` → registers；
4. A row-major / B col-major 布局；
5. fragment 到普通矩阵坐标的映射；
6. FP16/BF16 输入、FP32 accumulator。

### 学习者亲手实现

- Inline PTX 或 CUDA 提供的底层接口；
- Shared Memory 布局；
- lane 到 fragment 元素映射；
- `ldmatrix.sync.aligned`；
- `mma.sync.aligned.m16n8k16`。

### 证据门槛

- 小矩阵逐元素对拍；
- 非零随机输入和易定位输入；
- sanitizer；
- SASS 必须出现 `LDSM` 与 `HMMA`/目标 MMA 指令；
- 无 `LDL/STL` spill，或明确记录 spill 结果。

### 晋级条件

脱离完整 Attention，可以独立画出一次 MMA 的 lane/fragment/矩阵坐标映射并通过验证。

## M5：Tensor Core QK

### 为什么做

只替换 $QK^T$，保持 Softmax 和 PV 为已有 SIMT 路径，以单变量方式接入 Tensor Core。

### 学习者亲手实现

- Q/K Shared Memory layout；
- K 的逻辑转置或物理转置策略；
- Warp 对 Score MMA tile 的所有权；
- Score accumulator fragment 转换为 Softmax 可消费布局；
- scale 与 causal mask。

### 证据门槛

- QK Score tile 独立对拍，再接完整 Output 对拍；
- M3 的同 dtype、FP32-accumulate SIMT QK 作为性能单变量基线；
- 原 FP32 SIMT QK 继续作为数值 reference，但不替代同 dtype 性能基线；
- ncu 观察 Tensor pipe、eligible warps、Shared Memory；
- SASS 验证 `LDSM` 与 `HMMA`；
- 记录 fragment 转储/重排成本。

### 晋级条件

Tensor Core QK 的完整 Attention 输出正确，且能分离 MMA 收益与 fragment 重排成本。

## M6：Tensor Core PV

### 为什么做

完成 Attention 的第二个矩阵乘法 $PV$，处理 Softmax FP32 权重到低精度 MMA 输入的转换与布局。

### 学习者亲手实现

- FP32 Softmax weights 转 FP16/BF16；
- Probability tile 的 Shared Memory/fragment 布局；
- V fragment 加载；
- PV FP32 accumulator；
- Online rescale 与 MMA accumulator 的结合。

### 重点风险

- Probability 量化误差；
- QK 与 PV 对 tile 方向的不同要求；
- Probability 写回 Shared Memory 的成本；
- fragment 重排和 barrier 增加。

### 证据门槛

- 比较 Tensor-QK + SIMT-PV 与 Tensor-QK + Tensor-PV；
- 长序列误差与极端 Softmax；
- SASS 同时证明两段 MMA；
- ncu 分析 Tensor pipe、Shared Memory 和 short scoreboard。

### 晋级条件

QK 和 PV 都走目标 Tensor Core path，Output 正确且量化误差有系统记录。

## M7：Shared Memory Swizzle

### 为什么做

在 Tensor Core 数据流正确后，减少 `ldmatrix` 和 Shared Memory 访问的 bank conflict。

### 单变量实验顺序

1. 无 swizzle baseline；
2. 简单 padding；
3. XOR swizzle；
4. 分别处理 Q/K/V/Probability layout；
5. 保持 MMA tile、dtype 和 pipeline 不变。

### 学习者亲手实现

- logical → physical 地址映射；
- 逆映射与边界；
- 16B 对齐、`ldmatrix` 对齐和 stage stride；
- 不同 operand 的 swizzle 选择。

### 证据门槛

- 同一输入与同一 MMA path correctness；
- bank conflicts、wavefronts、short scoreboard；
- SASS 中 `LDSM` 路径不退化；
- 正常墙钟决定是否保留 swizzle。

### 晋级条件

能够从 lane 访问地址计算 bank，并用 ncu/SASS 证明 swizzle 的实际影响。

## M8：Multi-stage Pipeline

### 为什么做

将 Tensor Core 计算与 K/V global-to-shared 搬运重叠，实验 2/3/4 stages 的延迟隐藏与 residency 代价。

### 学习者亲手实现

- prologue、steady state、epilogue；
- current/next stage 状态；
- acquire/commit/wait/release；
- stage reuse 和 tail；
- 不同 stage 数的编译期配置。

### 证据门槛

- stage 复用专项 racecheck/synccheck；
- SASS 验证 Async 指令；
- 比较 2/3/4 stages 的 SMEM、waves、occupancy 和 long scoreboard；
- 至少包含一个改善和一个回退 shape。

### 晋级条件

能够根据 shape 和资源解释 stage 数选择，而不是固定宣称 stage 越多越好。

## M9：FlashAttention-2-style Mapping

### 为什么做

减少非矩阵乘法工作、跨 Warp reduction 和同步，学习 FA2 的工作划分思想，而不是照抄某个实现。

### 研究重点

- 沿 Query sequence 维度增加并行；
- Warp 间分配 Q rows；
- 每 Warp 独立维护输出片段；
- 减少 Shared Memory 中间结果；
- 降低 Online Softmax rescale 和跨 Warp通信成本。

### 证据门槛

- 与 M8 相同 dtype/tile/pipeline 的单变量对照；
- barriers、Warp stall、non-MMA instruction 比例；
- Tensor Core utilization 与正常墙钟；
- 明确与官方 FlashAttention-2 的相同点和不同点。

### 晋级条件

能从线程映射和 profiler 证据解释同步/非 MMA 工作的变化，不以名称宣称等同生产级 FA2。

## M10：Batch / Head / GQA / Varlen

这一阶段按子门槛顺序完成，不同时加入四项功能。

### M10a：Batch + Multi-head

- 输入扩展到 `[B,H,N,D]`；
- grid 显式映射 batch、head、query tile；
- correctness 覆盖不同 `B/H` 和 stride。

### M10b：MQA/GQA

- `Hq > Hkv`；
- `query_head → kv_head` 映射；
- K/V 在 Query heads 间复用；
- benchmark 绑定 `Hq/Hkv`。

### M10c：Variable-length

- cumulative sequence lengths；
- 每个 sequence 独立 causal 边界；
- 空序列、短序列和负载不均衡；
- 不能读取 padding 之外的 K/V。

### M10d：Bias/Fusion（可选）

- ALiBi 或 additive bias 单独加入；
- RoPE/QKV fusion 只在有清晰 IO 收益假设时进行；
- 每次只加入一种 fusion。

### 证据门槛

每个子阶段都有独立 correctness、sanitizer 和 benchmark；最终与 PyTorch SDPA 或适当 vendor path 比较，但明确 dtype/layout/功能差异。

### 晋级条件

M10a、M10b、M10c 全部完成；M10d 可选，不阻塞 Backward。

## M11：Backward

### 为什么做

完成训练侧 Attention 的梯度数据流，学习重计算、跨 CTA 梯度归约与数值稳定性。

### 数学范围

- $dQ$；
- $dK$；
- $dV$；
- Softmax backward；
- Forward 保存或重计算 log-sum-exp。

Dropout backward 延后，先完成无 dropout 版本。

### 学习者亲手实现

- backward 数学推导；
- dQ 与 dK/dV 的线程所有权；
- 多 CTA 对 dK/dV 的归约；
- atomic、split reduction 或两阶段归约对照；
- forward state 保存与重计算策略。

### 证据门槛

- PyTorch autograd/vendor reference；
- finite difference 小 shape 检查；
- 极端 logits 和长序列；
- sanitizer；
- forward + backward 总墙钟和额外 workspace；
- 原子竞争、reduction 和重计算的 ncu 证据。

### 晋级条件

$dQ/dK/dV$ 全部通过 reference 和数值梯度检查，训练侧性能证据完整。

---

# 主线 B：Inference Attention

## 为什么独立规划 Decode

Prefill 通常有多条 Query，适合矩阵乘法与二维 tiling；Decode 每次通常只有一个或少量 Query，核心瓶颈转向 KV Cache 带宽、并行归约、分页访问和动态调度。

```text
Prefill：Q=[N,D]，主要复用和 Tensor Core 问题
Decode：Q=[1,D]，KV Cache=[T,D]，主要带宽和并行切分问题
```

Decode 路线使用独立 runner、reference 和 benchmark 协议；不复用 Prefill 的性能结论。

## D1：Decode Baseline

### 目标

实现单 batch、单 Query head、连续 KV Cache 的直接 Decode Attention：

```text
Q：[D]
K/V Cache：[T,D]
Output：[D]
```

### 学习者亲手实现

- Q·K Score；
- Stable Softmax；
- Probability·V；
- causal 语义（Decode 通常只看已存在 Cache）；
- FP32 baseline。

### 证据门槛

- `T=1/2/15/16/17/127/128/1024`；
- `D=64/128`；
- CPU double reference；
- sanitizer；
- 记录 bytes/token 与 latency/token。

### 晋级条件

正确性和带宽基线完成，能够用 Roofline/算术强度解释瓶颈。

## D2：KV Cache + Online Softmax

### 目标

按三个单变量子门槛推进，不在一轮实验中同时改变数据流、dtype 和 Cache layout。

### D2a：FP32 Continuous Cache + Online Softmax

- 保持 D1 的 FP32 与连续 Cache layout；
- Decode 版 Online Softmax；
- KV tile tail；
- 不物化完整 Scores/Probability；
- 流式维护 `m/l/O_acc`。

### D2b：FP16/BF16 Cache

- 保持 D2a 的 tile 和线程映射；
- Cache 改为 FP16/BF16，Softmax state 与 accumulator 保持 FP32；
- 分开记录两种 dtype 的误差、带宽和容量。

### D2c：Cache Layout 对照

- 保持选定 dtype 与 Online Softmax 不变；
- 单独比较 token-major、head-major 或其他候选 layout；
- 明确每种 layout 的 vectorized load 和 head mapping 条件。

### 证据门槛

- 与 Materialized Decode 对拍；
- workspace 降为零或明确固定大小；
- HBM bytes、L2 hit、正常延迟；
- 长 context 数值稳定性。

### 晋级条件

D2a、D2b、D2c 分别完成独立 correctness 与性能对照；长 context 下正确稳定，并用 profiler 证明 D2a 的中间矩阵流量消除。

## D3：Warp-level Decode

### 目标

设计一个或多个 Warp 协作一条 Decode Query，优化 dot、Softmax 和 PV reduction。

### 实验方向

- 一个 Warp/query；
- 多 Warp/query；
- 每 lane 处理 feature 或 token；
- Warp shuffle reduction；
- Vectorized KV load。

### 证据门槛

- full-mask Warp collective 安全；
- 不同 `T/D` 的映射边界；
- registers、active/eligible warps、memory throughput；
- 至少保留一种负映射结果。

### 晋级条件

能够为不同 `T/D` 选择映射，并以证据说明选择原因。

## D4：MQA / GQA

### 目标

支持多个 Query heads 共享较少的 KV heads，连接现代 LLM 推理中的模型结构与 Kernel 映射。

### 学习者亲手实现

- `Hq → Hkv` 映射；
- 多 Query heads 的 grid/CTA 切分；
- KV 在 Query heads 间的 Cache/L2 复用；
- MHA、GQA、MQA 统一接口。

### 证据门槛

- `Hq/Hkv = 1/1, 8/8, 8/2, 8/1`；
- KV Cache 容量和 bytes/token；
- 正确性和 head mapping 边界；
- L2/DRAM throughput 与 latency/token。

### 晋级条件

三种 Attention head 模式统一验证，容量和性能结果可解释。

## D5：Split-KV

### 为什么做

当 Query 数很少时，一个 CTA/query 并行度不足。Split-KV 将 context 切给多个 CTA，再合并局部 Online Softmax 状态。

### 学习者亲手实现

- context split；
- 每个 split 的局部 `m_s/l_s/O_acc,s`，其中：

$$
O_{acc,s}=\sum_{j\in s} e^{x_j-m_s}V_j
$$

- `O_acc,s` 是未归一化分子，不是 split 内已经除以 `l_s` 的最终输出；
- 跨 split 合并：

$$
m=\max_s m_s
$$

$$
l=\sum_s e^{m_s-m}l_s
$$

$$
O=\frac{\sum_s e^{m_s-m}O_{acc,s}}{l}
$$

- 两阶段 reduction 或 cooperative 策略；
- split 数 dispatch。

### 证据门槛

- split `1/2/4/8`；
- 非整除 context；
- 合并数值稳定性；
- workspace、额外 launch 与并行收益；
- 不同 workload/shape 的最佳 split 边界，至少绑定 `B/Hq/T/D/dtype/GPU`；
- 记录 workspace 和第二阶段 reduction 成本。

### 晋级条件

合并公式正确，并形成 `(B,Hq,T,D,dtype,GPU) → split count` 的证据化 dispatch 规则；不把单 Query 结果泛化为所有负载。

## D6：PagedAttention

### 目标

将连续 KV Cache 改为固定大小 page/block，通过 block table 访问非连续物理内存。

### 数据结构

- logical token position；
- physical block ID；
- block table；
- page size；
- 最后一个不完整 page；
- 多 sequence 独立表。

### 学习者亲手实现

- logical → physical 地址翻译；
- 跨 page KV tile；
- page tail；
- GQA + Paged KV Cache；
- contiguous cache fallback/reference。

### 证据门槛

- 随机 block table；
- 非连续 page、page tail、不同 page size；
- 非法映射防护；
- correctness/sanitizer；
- L2 hit、address calculation、latency/token；
- 与连续 KV Cache 的开销对照。

### 晋级条件

随机分页布局下稳定正确，并能解释相对连续 Cache 的开销与服务收益。

## D7：Continuous Batching

### 目标

从单请求 Kernel 扩展到不同 context length、不同 page table 的动态请求批次，学习调度与 Kernel 形状之间的关系。

### 研究范围

- request metadata；
- 不同 `T` 的请求混合；
- work queue 或 grid dispatch；
- padding batching 与 continuous batching 对照；
- tail latency、吞吐和公平性；
- 不实现完整推理服务器，但建立可控模拟器。

### 证据门槛

- 多请求 correctness；
- 固定且版本化的 arrival trace、context length 分布、请求速率、并发度、随机种子、预热请求数和正式采样请求数；
- CUDA Event 单独记录 Kernel/device 时间；
- monotonic host clock 记录包含排队、CPU 调度、提交和 GPU 完成的端到端 latency；
- 分别报告 tokens/s、平均 latency、P50/P95/P99；
- active SM、负载不平衡和 wave；
- 明确 Kernel 时间与调度时间边界。

### 晋级条件

能够从模型结构、KV Cache、Kernel 映射和调度四层解释一个简化 LLM serving Attention 路径，并能从固定 arrival trace 复现端到端 tail latency。

---

# 里程碑状态表

| ID | 里程碑 | 状态 | 依赖 |
| --- | --- | :---: | --- |
| B0 | FP32 Prefill Baseline | ✅ | - |
| M1 | `Br=4` Query-tiled FP32 SIMT | ⬜ | B0 |
| M2 | Warp-per-query | ⬜ | M1 |
| M3 | FP16/BF16 SIMT | ⬜ | M2 |
| M4 | MMA Playground | ⬜ | M3 |
| M5 | Tensor Core QK | ⬜ | M4 |
| M6 | Tensor Core PV | ⬜ | M5 |
| M7 | Shared Memory Swizzle | ⬜ | M6 |
| M8 | Multi-stage Pipeline | ⬜ | M7 |
| M9 | FA2-style Mapping | ⬜ | M8 |
| M10 | Batch/Head/GQA/Varlen | ⬜ | M9 |
| M11 | Backward | ⬜ | M10 |
| D1 | Decode Baseline | ⬜ | M11 |
| D2 | KV Cache + Online Softmax | ⬜ | D1 |
| D3 | Warp-level Decode | ⬜ | D2 |
| D4 | MQA/GQA | ⬜ | D3 |
| D5 | Split-KV | ⬜ | D4 |
| D6 | PagedAttention | ⬜ | D5 |
| D7 | Continuous Batching | ⬜ | D6 |

状态只在证据门槛完成后更新。开始某个里程碑时，再为该里程碑写独立设计和实现计划；本路线图不提前创建代码目录或空 Kernel。
