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

# 非 canonical 的最小 evidence smoke
AP_M1_SHAPES=128x64 AP_M1_CAUSAL=0 \
AP_M1_WARMUP=0 AP_M1_ITERATIONS=1 AP_M1_REPEATS=1 \
AP_M1_ALLOW_DIRTY=1 \
projects/attention_prefill/scripts/benchmark_m1.sh
```

Evidence runner 支持 `--metadata-only`，输出编译期 source fingerprint、带完整 payload hash 的 build contract，以及实际 CUDA device 的 index、UUID、名称 token、SM 和 driver version。Canonical benchmark 必须无位置参数运行，只接受非 symlink 的标准 runner 普通文件；smoke 才允许传入自定义 runner。脚本按确定性的 filesystem manifest 重算当前 source fingerprint，并独立解析、逐项验证 CMake 生成的 build attestation；正式协议固定为 `release-sm80-<hash>`、A100 80GB、SM 8.0。详细 provenance、威胁模型和 CSV 合同见 [results/README.md](results/README.md)。

M1 profile 与 SASS 同样区分 canonical 和 smoke。标准 evidence runner、当前 source fingerprint、canonical build attestation、三组固定 canonical shape 与 official 输出目录全部匹配时，才会生成正式文件；自定义 runner、非 canonical shape 和 smoke override 必须写入路径名含 `smoke` 的独立目录。ncu 合同只支持 `2026.2.*`，其他版本一律 fail closed。

当前 M1 已完成 correctness、full sanitizer、40 行 CUDA Event canonical benchmark、六点 Nsight Compute 和 SASS 证据。Br4 在 `N<=512` 回退，在 `N>=1024` 全部获益；最大收益为 `2048x64 causal=1` 的 `1.489x`。完整结果见 [results/generated/a100-fp32-m1.md](results/generated/a100-fp32-m1.md)、[results/evidence/m1-ncu-summary.md](results/evidence/m1-ncu-summary.md)和[results/evidence/m1-sass.md](results/evidence/m1-sass.md)。下一阶段为 M2 Warp-per-query。
