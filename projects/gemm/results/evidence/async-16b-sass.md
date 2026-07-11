# async-16b SASS 静态证据

由 `projects/gemm/scripts/extract_sass.sh async-16b` 从当前构建的 `gemm_runner` 生成。计数是目标函数内的静态指令数，不代表运行时执行次数。

| Opcode | 静态数量 |
| --- | ---: |
| `LDG*` | 0 |
| `LDG.E.128` | 0 |
| `LDGSTS*` | 4 |
| `LDGSTS.E.BYPASS.128` | 4 |
| `LDGSTSBAR*` | 2 |
| `FFMA` | 192 |
| `LDL*` | 0 |
| `STL*` | 0 |

`LDL`/`STL` 是 local-memory spill 的静态风险信号；最终仍需结合 ptxas 和 profiler 判断。

## 宽加载与异步搬运片段

```text
        /*0650*/                   LDGSTS.E.BYPASS.128 [R7], [R4.64] ;                     /* 0x0000000004077fae */
        /*0830*/                   LDGSTS.E.BYPASS.128 [R7+0x2000], [R4.64] ;              /* 0x0200000004077fae */
        /*0930*/                   ARRIVES.LDGSTSBAR.64 [URZ+0x4000] ;                     /* 0x00400000ff0079b0 */
        /*1160*/                   LDGSTS.E.BYPASS.128 [R17], [R8.64] ;                    /* 0x0000000008117fae */
        /*13d0*/                   LDGSTS.E.BYPASS.128 [R17], [R8.64] ;                    /* 0x0000000008117fae */
        /*14f0*/                   ARRIVES.LDGSTSBAR.64 [UR13] ;                           /* 0x00000000ff0079b0 */
```
