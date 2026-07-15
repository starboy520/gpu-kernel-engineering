# Attention Prefill M1 Nsight Compute 摘要 Smoke（单点诊断，非六点 canonical）

> ncu duration 仅用于 profiler 内部对照，不等于 CUDA Event 或端到端 wall-clock；单位来自 ncu CSV，不静默换算。

> Evidence kind：`smoke`；当前只包含一个 Br1/Br4 pair，不是正式六点 canonical 结论。

输入：`projects/attention_prefill/results/profiles/smoke/br1-256x64-causal0-smoke-metrics.csv`、`projects/attention_prefill/results/profiles/smoke/br4-256x64-causal0-smoke-metrics.csv`

## 环境与构建身份

| Field | Value |
| --- | --- |
| Git commit | `880786b58b21354a32c9ebdcfe517b85bd9d9feb` |
| Runner SHA-256 | `c84b3f158868075bb604f9645bc46f778dd216a799d32bf0e3f8d8a6e3c922f6` |
| Source SHA-256 | `9c0240662d12ed27f5e4eb660727d521737400a5e5fe46e99a26da4c99cc4372` |
| Build contract | `release-sm80-81b0720deaaf5ea6` |
| Build payload SHA-256 | `81b0720deaaf5ea63ec6952b6afbc5ca01d3428b4ec27d328bf940383986a41a` |
| Device index | `0` |
| GPU UUID | `GPU-6b24bab9-c81f-0e0e-ff68-a11dcf17018b` |
| GPU | `NVIDIA_A100_80GB_PCIe` |
| SM | `8.0` |
| CUDA driver | `13030` |
| Nsight Compute | `2026.2.0.0` |
| Profile timestamps | `2026-07-15T11:55:14Z, 2026-07-15T11:57:17Z` |

## 256x64 causal=0

Kernel：Br1 `<unnamed>::tiled_attention_kernel(const float *, const float *, const float *, float *, int, int, bool)`；Br4 `<unnamed>::query_tiled_kernel(const float *, const float *, const float *, float *, int, int, bool)`。

| Metric | Br1 | Br4 | Br4 / Br1 |
| --- | ---: | ---: | ---: |
| Block | (128, 1, 1) | (128, 1, 1) | - |
| Grid | (256, 1, 1) | (64, 1, 1) | - |
| ncu duration | 130.688000 us | 207.424000 us | 1.587x |
| Registers/thread | 31 register/thread | 34 register/thread | 1.097x |
| Static SMEM/block | 17.484000 Kbyte/block | 20.784000 Kbyte/block | 1.189x |
| Waves/SM | 0.26 | 0.08 | 0.308x |
| SMEM limit blocks/SM | 9.000000 block | 7.000000 block | 0.778x |
| Register limit blocks/SM | 16.000000 block | 12.000000 block | 0.750x |
| Achieved occupancy | 14.795274 % | 6.249886 % | 0.422x |
| Eligible warps/cycle | 0.131465 warp | 0.063928 warp | 0.486x |
| Long scoreboard | 10.214092 inst | 7.776536 inst | 0.761x |
| Short scoreboard | 0.636098 inst | 1.509914 inst | 2.374x |
| Barrier stall | 4.575559 inst | 2.156947 inst | 0.471x |
| Warp latency | 19.964861 cycle | 15.642526 cycle | 0.784x |
| SM throughput | 11.653652 % | 3.749374 % | 0.322x |
| DRAM throughput | 0.083489 % | 0.055906 % | 0.670x |
| L2 throughput | 2.126381 % | 0.809980 % | 0.381x |
| L1 throughput | 13.945472 % | 6.597232 % | 0.473x |
| DRAM read bytes | 211.072000 Kbyte | 224.384000 Kbyte | 1.063x |
| DRAM write bytes | 0 byte | 0 byte | - |
| L2 read sectors | 453358.000000 sector | 277085.000000 sector | 0.611x |
| L2 write sectors | 3204.000000 sector | 4775.000000 sector | 1.490x |
| L2 sector hit rate | 97.382309 % | 94.103410 % | 0.966x |
| Global load sectors | 1050624.000000 sector | 264192.000000 sector | 0.251x |
| Global load requests | 262656 | 66048 | 0.251x |
| Global store sectors | 2048.000000 sector | 2048.000000 sector | 1.000x |
| Global store requests | 512 | 512 | 1.000x |
| Shared bank conflicts | 917504 | 960512 | 1.047x |
| Shared wavefronts | 1791488 | 1496064 | 0.835x |
