# FP32 GEMM 优化阶梯

**状态：** 正在重写

## 范围

在 NVIDIA A100（`sm_80`）上实现 row-major 的 FP32 GEMM：$C=A\times B$。

当前只做 CUDA Core 版本，不包含 Tensor Core、转置输入和 batched GEMM。

## 实现顺序

1. Naive
2. Shared memory tiling
3. 2D register tiling
4. `float4` 向量化加载
5. `cp.async` 双缓冲
6. cuBLAS pedantic FP32 基线

所有版本共用同一套输入、CPU 对拍和计时框架。性能数据会在完整正确性测试和 sanitizer 通过后重新采集，不沿用学习仓库里的旧结果。