# Tensor Core GEMM Results

当前 G1/G1.5 只验证机制、tiling 和机器路径，不发布正式性能结论。

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

G1.5 Direct 额外记录：

- A row-major / B col-major / C row-major；
- logical shape 与 padded shape；
- K=16/32/48 的 accumulator 生命周期；
- M/N 多 Tile 与非整除 padding correctness；
- Direct SASS 中的 `HMMA`、`LDG/STG`、`MOVM` 与 `LDL/STL`。

当前 Direct SASS 静态证据显示 `HMMA=10`、`MOVM=0`、`LDL/STL=0`。静态数量受编译器循环展开影响，不等于任意 shape 的 runtime 指令次数；`MOVM=0` 只描述当前 A100、工具链与 B col-major 构建。

G4/G5 才开始内部 smoke benchmark；正式 cuBLAS 和大矩阵 canonical 结果留待后续决定。
