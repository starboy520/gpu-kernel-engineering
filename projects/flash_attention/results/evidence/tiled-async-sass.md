# tiled-async SASS 静态证据

计数来自目标函数的静态 SASS，不代表运行时执行次数。

| Opcode | 静态数量 |
| --- | ---: |
| `LDGSTS*` | 4 |
| `LDGSTS.E.BYPASS.128` | 4 |
| `ARRIVES.LDGSTSBAR*` | 2 |
| `LDL*` | 0 |
| `STL*` | 0 |

## 异步搬运片段

```text
        /*0940*/                   LDGSTS.E.BYPASS.128 [R7+0x200], [R2.64] ;                    /* 0x0020000002077fae */
        /*0c70*/                   LDGSTS.E.BYPASS.128 [R5+0x4200], [R2.64] ;                   /* 0x0420000002057fae */
        /*0d70*/                   ARRIVES.LDGSTSBAR.64 [URZ+0x8450] ;                          /* 0x00845000ff0079b0 */
        /*15a0*/               @P0 LDGSTS.E.BYPASS.128 [R13], [R8.64] ;                         /* 0x00000000080d0fae */
        /*18b0*/               @P0 LDGSTS.E.BYPASS.128 [R13], [R8.64] ;                         /* 0x00000000080d0fae */
        /*19d0*/                   ARRIVES.LDGSTSBAR.64 [UR6] ;                                 /* 0x00000000ff0079b0 */
```
