# Attention Prefill M1 SASS 静态证据

> 下表是目标函数的静态指令数量；静态数量不等于 runtime 执行次数。

| Metadata | Value |
| --- | --- |
| Evidence kind | `canonical` |
| Git commit | `e6ce1d08ce25297127ced33e18598beca366a68e` |
| Binary | `build/projects/attention_prefill/attention_prefill_evidence_runner` |
| Binary SHA-256 | `0b701c87c86c9b315af35eeedf5826d66313648061ce2b186724a7485069a928` |
| Source fingerprint | `141a5daef9d78a3ba79b2e6d39e7db95eb3c82baf9117b255ac3fabd277b6040` |
| Build contract | `release-sm80-81b0720deaaf5ea6` |
| Build payload SHA-256 | `81b0720deaaf5ea63ec6952b6afbc5ca01d3428b4ec27d328bf940383986a41a` |
| Device index | `0` |
| GPU UUID | `GPU-6b24bab9-c81f-0e0e-ff68-a11dcf17018b` |
| GPU / SM | `NVIDIA_A100_80GB_PCIe` / `8.0` |
| CUDA driver | `13030` |
| Full SASS directory | `projects/attention_prefill/results/sass` |
| Br1 signature | `tiled_attention_kernel(float const*, float const*, float const*, float*, int, int, bool)` |
| Br4 signature | `query_tiled_kernel(float const*, float const*, float const*, float*, int, int, bool)` |

| Opcode | Br1 | Br4 |
| --- | ---: | ---: |
| `FFMA` | 114 | 140 |
| `BAR` | 6 | 6 |
| `LDG` | 3 | 7 |
| `STG` | 1 | 5 |
| `LDS` | 87 | 98 |
| `STS` | 33 | 72 |
| `LDL` | 0 | 0 |
| `STL` | 0 | 0 |
| `HMMA` | 0 | 0 |
| `LDGSTS` | 0 | 0 |

M1 ISA 合同：`HMMA=0` 且 `LDGSTS=0`；两条路径 `FFMA>0`。

**Spill warning: none**
