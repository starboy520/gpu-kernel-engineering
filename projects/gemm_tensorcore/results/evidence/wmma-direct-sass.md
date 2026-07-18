# G1.5 Direct WMMA SASS

> 静态 opcode 数量不等于 runtime 执行次数。Direct 版本使用 Global → fragment，不包含 Shared Memory 或异步复制。

| Opcode | Count | 解释 |
| --- | ---: | --- |
| `HMMA` | 10 | Tensor Core matrix multiply-accumulate |
| `LDG` | 40 | Global input load |
| `STG` | 4 | Global output store |
| `MOVM` | 0 | Matrix operand register rearrangement |
| `LDL` | 0 | Local-memory load / spill evidence |
| `STL` | 0 | Local-memory store / spill evidence |
