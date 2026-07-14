# Advanced Attention Prefill Roadmap

> 原则：每次只改变一个核心变量；先证明正确和安全，再谈性能。

## 当前进度

```text
M1 Kernel correctness/safety  [完成]
M1 benchmark/ncu/SASS         [下一步]
M2 Warp-per-query             [未开始]
M3 及以后                     [未开始]
```

## 学习路线

| 阶段 | 实现内容 | 重点学习 | 完成证据 |
| --- | --- | --- | --- |
| M1 Query-tiled FP32 SIMT | 一个 CTA 处理 4 条 Query，共享一个 K/V tile | 二维 tile、逐行 Online Softmax、全局/局部索引、三种 tail | correctness、sanitizer、benchmark、ncu、SASS |
| M2 Warp-per-query | 一个 Warp 固定负责一条 Query | Warp ownership、lane 到 feature 映射、寄存器 `O_acc`、减少 CTA barrier | 与 M1 同矩阵对拍；比较 barrier、register、occupancy |
| M3 FP16/BF16 SIMT | 低精度 Q/K/V，FP32 Score 与输出累加 | `half2`/BF16、向量化加载、数值误差、fallback | dtype 独立容差；SASS 验证 load/convert/FP32 FMA |
| M4 MMA Playground | 脱离 Attention 做单 MMA 和多 K-step | `m16n8k16`、fragment、lane 映射、`ldmatrix`、`mma.sync` | 小矩阵逐元素对拍；SASS 出现 `LDSM`/`HMMA` |
| M5 Tensor Core QK | 只把 $QK^T$ 换成 Tensor Core | Q/K Shared 布局、K 转置、Score fragment 导出 | QK tile 和完整输出对拍；分离 MMA 收益与重排成本 |
| M6 Tensor Core PV | 再把 $PV$ 换成 Tensor Core | Softmax 权重降精度、Probability/V fragment、FP32 accumulator | PV 与完整 Attention 对拍；记录转换和布局成本 |
| M7 Shared Memory Swizzle | 调整 Q/K/V/Probability 布局 | Bank Conflict、padding、swizzle、地址变换 | ncu 证明 bank conflict 改变；记录资源代价 |
| M8 Multi-stage Pipeline | 多 stage 搬运与计算重叠 | `cp.async`、stage 生命周期、producer/consumer ownership | sanitizer；ncu stall；SASS 出现目标异步复制指令 |
| M9 FA2-style Mapping | 重构 CTA/Warp 工作划分 | 降低非矩阵 FLOP、减少同步、提高并行度 | 与前版同协议 benchmark；解释收益来源 |
| M10 工程化 Shape | Batch、Head、GQA、Varlen | tensor layout、dispatch、head mapping、变长序列边界 | 多 shape/dtype/causal 矩阵和 vendor reference |
| M11 Backward | 实现 Attention 反向传播 | 重计算、梯度公式、数值稳定、原子与归约策略 | 与框架 autograd 对拍；完整 sanitizer 和性能证据 |

## 每个阶段固定流程

```text
数学公式与数据流
→ 线程/数据所有权
→ 先准备失败测试
→ CPU 或 vendor reference 对拍
→ tail 与特殊输入
→ 四类 Compute Sanitizer
→ CUDA Event benchmark
→ Nsight Compute 验证瓶颈
→ PTX/SASS 验证关键指令
→ 记录正收益、负收益与下一假设
```

## M1 接下来具体做什么

1. 固定 canonical benchmark shape 与计时协议。
2. 对比冻结的 `Br=1` FP32 baseline 与当前 `Br=4` 实现。
3. 用 ncu 检查 K/V 请求加载、L1/L2、barrier、long scoreboard、Shared Memory 和 occupancy。
4. 用 SASS 确认当前仍是 FP32 SIMT 标量 FMA 路径。
5. 证据归档后再进入 M2；不要现在就修改阶段 E 的线程映射。

一句话记忆：

> M1 学二维 Query tile，M2 学 Warp ownership，M3 学低精度，M4 先单练 MMA，M5/M6 再把 Tensor Core 接回 Attention。
