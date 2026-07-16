# M1 Query-tiled vs M2 Warp-per-query SASS（开发期）

> 静态 opcode 数量不等于 runtime 执行次数；正式发布前需在 clean commit 上重采。

| Opcode | M1 | M2 |
| --- | ---: | ---: |
| `FFMA` | 140 | 102 |
| `SHFL` | 0 | 78 |
| `BAR` | 6 | 2 |
| `LDG` | 7 | 31 |
| `STG` | 5 | 4 |
| `LDS` | 98 | 52 |
| `STS` | 72 | 2 |
| `LDL` | 0 | 0 |
| `STL` | 0 | 0 |
| `HMMA` | 0 | 0 |
| `LDGSTS` | 0 | 0 |
