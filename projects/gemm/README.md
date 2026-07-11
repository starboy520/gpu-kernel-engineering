# FP32 GEMM 优化阶梯

**状态：** 正在重写

## 范围

在 NVIDIA A100（`sm_80`）上实现 row-major 的 FP32 GEMM：$C=A\times B$。

当前只做 CUDA Core 版本，不包含 Tensor Core、转置输入和 batched GEMM。

## 实现顺序

1. [Naive](docs/naive.md)
2. [Shared memory tiling](docs/shared-tiled.md)
3. [2D register tiling](docs/register-tiled.md)
4. [`float4` 向量化加载](docs/vectorized.md)
5. [`cp.async` 双缓冲](docs/async-pipeline.md)
6. [cuBLAS pedantic FP32 基线](docs/cublas-baseline.md)

所有版本共用同一套输入、CPU 对拍和计时框架。性能数据会在完整正确性测试和 sanitizer 通过后重新采集，不沿用学习仓库里的旧结果。

## 验收流程

每完成一个 kernel，都按照 [CUDA GEMM Kernel 验收手册](docs/kernel-verification-guide.md) 逐步执行编译、对拍、sanitizer、回归测试和 benchmark。