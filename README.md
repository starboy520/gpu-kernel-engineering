# GPU Kernel Engineering

这个仓库记录我手写 CUDA kernel 和做性能优化的过程。

每个项目都从能对拍的基础版本开始，再根据 benchmark、Nsight Compute 和 SASS 结果判断下一步怎么改。README 里只保留当前代码能够复现的数据，失败或没有收益的优化也会如实记录。

## 项目

- [FP32 GEMM 优化阶梯](projects/gemm/)

## 后续计划

- FlashAttention
- CUDA 常用算子

这两项目前还没有放进仓库，等 GEMM 项目完成后再继续。