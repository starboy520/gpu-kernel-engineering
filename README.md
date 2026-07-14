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

Async 16B 没有超过 Vectorized，但仍作为负结果保留：`cp.async` 降低了 long-scoreboard stall，代价是更高的 short-scoreboard stall 和 shared-memory bank conflict，最终墙钟性能回退约 4.8%。

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

## Featured Project：FP32 FlashAttention 数据流重建

第二个作品的当前教育版范围已经完成：

- Naive Materialized Attention：`QK^T → Stable Softmax → PV`；
- Online Tiled Attention：K/V 分块、running `m/l/O_acc`、causal 和尾块；
- Warp 并行归约：正确性与 sanitizer 通过，但没有稳定墙钟收益，作为负结果保留；
- K/V `cp.async` 双缓冲：硬件异步指令、正确性与 sanitizer 已验证，墙钟收益依赖 shape；
- CPU double reference、统一 runner、CTest correctness 与 Compute Sanitizer 验证；
- Tiled 路径不分配完整 `N×N` workspace。

当前版本是单 batch、单 head、FP32 forward educational/research baseline。48 行 A100 canonical benchmark 已完成：non-causal `D=128` 下 Async 在 `N=512` 延迟增加 18.2%、`N=768` 近似持平、`N=1024` 延迟降低 17.6%；`D=64` 在 `N=512/768` 明显回退，`N=1024` 仍仅近似持平。ncu 与 SASS 证明异步指令和 long-scoreboard 改善，但不宣称跨 shape 稳定加速。

- [项目状态、已完成证据与迭代路线](projects/flash_attention/)
- [Naive Materialized Kernel](projects/flash_attention/kernels/naive.cu)
- [Online Tiled Kernel](projects/flash_attention/kernels/tiled.cu)
- [Warp 并行归约 Kernel](projects/flash_attention/kernels/tiled_parallel.cu)
- [并行归约负结果分析](projects/flash_attention/docs/parallel-reduction.md)
- [`cp.async` 双缓冲实验](projects/flash_attention/docs/async-pipeline.md)
- [FlashAttention 性能实验方法](projects/flash_attention/docs/methodology.md)
- [Nsight Compute 实操手册](projects/flash_attention/docs/ncu-hands-on.md)
- [A100 canonical benchmark](projects/flash_attention/results/generated/a100-fp32.md)
- [Tiled correctness 入口](projects/flash_attention/scripts/test_tiled.sh)

## Roadmap

- FlashAttention：Naive、Online Tiled、Warp 并行归约、`cp.async`、canonical benchmark、ncu 与 SASS 证据已完成。
- Attention Lab：[里程碑路线图](ATTENTION_ROADMAP.md)按证据门槛继续 Advanced Prefill，再进入 Decode、Split-KV、PagedAttention 与 Continuous Batching。
- CUDA 常用算子：计划中，尚未实现。
