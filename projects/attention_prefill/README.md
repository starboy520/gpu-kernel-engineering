# Advanced Attention Prefill

Advanced Prefill 作品从 `Br=4` Query-tiled FP32 SIMT 开始，后续逐步加入 Warp ownership、FP16/BF16、Tensor Core、Shared Memory swizzle、multi-stage pipeline、FA2-style mapping 与 backward。

简版学习顺序与每阶段目标见 [ROADMAP.md](ROADMAP.md)。完整教材保存在同级 `cuda_study` 仓库。

当前阶段只实现 M1：

```text
Br=4
Bc=16
FP32 input / accumulation / output
single batch / single head
SIMT scalar FMA
forward only
```

核心 Kernel 由学习者亲自实现：[kernels/query_tiled.cu](kernels/query_tiled.cu)。工程提供 CPU double reference、最小 runner、126-case shape 矩阵、2 个 Online Softmax 特殊输入和公共 launcher 合同测试。

构建后运行：

```bash
build/projects/attention_prefill/attention_prefill_runner \
    --n 17 --d 65 --causal 1

projects/attention_prefill/tests/test_query_tiled.sh

projects/attention_prefill/scripts/sanitize.sh full
```

当前 M1 状态：correctness 与 safety 已完成；下一步采集 CUDA Event benchmark、Nsight Compute 与 SASS 证据。在这些证据完成前不发布性能结论。
