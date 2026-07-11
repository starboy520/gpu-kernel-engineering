# GPU Kernel Engineering

面向 GPU 性能工程的 CUDA kernel 作品集：从手写实现出发，用 correctness、benchmark、Nsight Compute 和 SASS 建立完整证据链。当前结果采集于 NVIDIA A100 80GB PCIe（`sm_80`），只展示仓库内可以复现的数据。

## Featured Project：FP32 GEMM

在 row-major `C = A × B` 上完成从 Naive 到 `float4` Vectorized、再到 `cp.async` 双缓冲的优化阶梯。正式 `2048³` 结果中，Vectorized 达到 **12.88 TFLOPS / 73.0% cuBLAS**。

| 阶段 | Latency | Throughput | 相对结果 |
| --- | ---: | ---: | --- |
| Naive | 4.697866 ms | 3.66 TFLOPS | 20.7% cuBLAS |
| Shared tiled | 3.222119 ms | 5.33 TFLOPS | 1.46× Naive |
| Register tiled | 2.644419 ms | 6.50 TFLOPS | 1.22× Shared |
| Vectorized | 1.333719 ms | 12.88 TFLOPS | 73.0% cuBLAS |
| Async 16B | 1.398333 ms | 12.29 TFLOPS | 69.6% cuBLAS |
| cuBLAS pedantic FP32 | 0.973251 ms | 17.65 TFLOPS | 100% |

Async 16B 没有超过 Vectorized，但仍作为负结果保留：`cp.async` 降低了 long-scoreboard stall，代价是更高的 short-scoreboard stall 和 shared-memory bank conflict，最终墙钟性能回退约 4.7%。

- [项目说明与优化阶梯](projects/gemm/)
- [canonical 三组 shape 结果表](projects/gemm/results/generated/a100-fp32.md)
- [实验方法与 profiler 解释](projects/gemm/docs/methodology.md)
- [Vectorized SASS：`LDG.E.128`](projects/gemm/results/evidence/vectorized-sass.md)
- [Async SASS：`LDGSTS.E.BYPASS.128`](projects/gemm/results/evidence/async-16b-sass.md)

## 方法

1. **Correctness**：统一输入与 reference，对齐 fast path / fallback 的实际执行路径。
2. **Safety**：用 Compute Sanitizer 检查越界、竞态、同步和未初始化访问。
3. **Benchmark**：固定环境与协议，多轮测量后取 latency 中位数。
4. **Explain**：用 ncu 和 SASS 验证优化是否生效，并解释收益或回退。

## 复现

以下命令均从仓库根目录执行，需要 CUDA Toolkit、CMake 3.25+；profiler 与 SASS 命令还需要 Nsight Compute CLI 和 `cuobjdump`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

projects/gemm/scripts/validate.sh
projects/gemm/scripts/sanitize.sh quick
projects/gemm/scripts/benchmark.sh

projects/gemm/scripts/profile.sh vectorized 2048 2048 2048
projects/gemm/scripts/extract_sass.sh vectorized
projects/gemm/scripts/extract_sass.sh async-16b
```

完整验收顺序、输出路径和自定义 runner 用法见 [GEMM 性能实验方法](projects/gemm/docs/methodology.md) 与 [Kernel 验收手册](projects/gemm/docs/kernel-verification-guide.md)。

## Roadmap

- FlashAttention：计划中，尚未实现。
- CUDA 常用算子：计划中，尚未实现。
