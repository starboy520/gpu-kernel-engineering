# Tensor Core GEMM Results

当前 G1 只验证机制，不发布正式性能结论。

结果分层：

```text
results/evidence/    可提交的紧凑 SASS 与阶段结论
results/sass/        本地完整反汇编，不提交
```

G1 必须记录：

- GPU、SM、CUDA 和 Git commit；
- FP16 input / FP32 accumulator/output；
- input pattern；
- correctness error；
- `HMMA` 是否生成；
- `LDL/STL` 是否出现。

G4/G5 才开始内部 smoke benchmark；正式 cuBLAS 和大矩阵 canonical 结果留待后续决定。
