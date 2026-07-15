# Attention Prefill M1 Nsight Compute 摘要

> ncu duration 仅用于 profiler 内部对照，不等于 CUDA Event 或端到端 wall-clock；单位来自 ncu CSV，不静默换算。

> Evidence kind：`canonical`；包含完整六点 canonical profile。

输入：`projects/attention_prefill/results/profiles/br1-256x64-causal0-metrics.csv`、`projects/attention_prefill/results/profiles/br4-256x64-causal0-metrics.csv`、`projects/attention_prefill/results/profiles/br1-1024x128-causal0-metrics.csv`、`projects/attention_prefill/results/profiles/br4-1024x128-causal0-metrics.csv`、`projects/attention_prefill/results/profiles/br1-1024x128-causal1-metrics.csv`、`projects/attention_prefill/results/profiles/br4-1024x128-causal1-metrics.csv`

## 环境与构建身份

| Field | Value |
| --- | --- |
| Git commit | `e6ce1d08ce25297127ced33e18598beca366a68e` |
| Runner SHA-256 | `0b701c87c86c9b315af35eeedf5826d66313648061ce2b186724a7485069a928` |
| Source SHA-256 | `141a5daef9d78a3ba79b2e6d39e7db95eb3c82baf9117b255ac3fabd277b6040` |
| Build contract | `release-sm80-81b0720deaaf5ea6` |
| Build payload SHA-256 | `81b0720deaaf5ea63ec6952b6afbc5ca01d3428b4ec27d328bf940383986a41a` |
| Device index | `0` |
| GPU UUID | `GPU-6b24bab9-c81f-0e0e-ff68-a11dcf17018b` |
| GPU | `NVIDIA_A100_80GB_PCIe` |
| SM | `8.0` |
| CUDA driver | `13030` |
| Nsight Compute | `2026.2.0.0` |
| Profile timestamps | `2026-07-15T13:14:16Z, 2026-07-15T13:22:30Z, 2026-07-15T13:25:02Z, 2026-07-15T13:27:05Z, 2026-07-15T13:29:08Z, 2026-07-15T13:31:10Z` |

## 256x64 causal=0

Kernel：Br1 `<unnamed>::tiled_attention_kernel(const float *, const float *, const float *, float *, int, int, bool)`；Br4 `<unnamed>::query_tiled_kernel(const float *, const float *, const float *, float *, int, int, bool)`。

| Metric | Br1 | Br4 | Br4 / Br1 |
| --- | ---: | ---: | ---: |
| Block | (128, 1, 1) | (128, 1, 1) | - |
| Grid | (256, 1, 1) | (64, 1, 1) | - |
| ncu duration | 130.624000 us | 207.104000 us | 1.585x |
| Registers/thread | 31 register/thread | 34 register/thread | 1.097x |
| Static SMEM/block | 17.484000 Kbyte/block | 20.784000 Kbyte/block | 1.189x |
| Waves/SM | 0.26 | 0.08 | 0.308x |
| SMEM limit blocks/SM | 9.000000 block | 7.000000 block | 0.778x |
| Register limit blocks/SM | 16.000000 block | 12.000000 block | 0.750x |
| Achieved occupancy | 14.795098 % | 6.249885 % | 0.422x |
| Eligible warps/cycle | 0.131362 warp | 0.063980 warp | 0.487x |
| Long scoreboard | 10.155812 inst | 7.750293 inst | 0.763x |
| Short scoreboard | 0.636446 inst | 1.509914 inst | 2.372x |
| Barrier stall | 4.613097 inst | 2.175282 inst | 0.472x |
| Warp latency | 19.952036 cycle | 15.629792 cycle | 0.783x |
| SM throughput | 11.656926 % | 3.741358 % | 0.321x |
| DRAM throughput | 0.083541 % | 0.055998 % | 0.670x |
| L2 throughput | 2.127566 % | 0.810866 % | 0.381x |
| L1 throughput | 13.949353 % | 6.583128 % | 0.472x |
| DRAM read bytes | 211.072000 Kbyte | 224.384000 Kbyte | 1.063x |
| DRAM write bytes | 0 byte | 0 byte | - |
| L2 read sectors | 453332.000000 sector | 276384.000000 sector | 0.610x |
| L2 write sectors | 3077.000000 sector | 3053.000000 sector | 0.992x |
| L2 sector hit rate | 97.415797 % | 95.484539 % | 0.980x |
| Global load sectors | 1050624.000000 sector | 264192.000000 sector | 0.251x |
| Global load requests | 262656 | 66048 | 0.251x |
| Global store sectors | 2048.000000 sector | 2048.000000 sector | 1.000x |
| Global store requests | 512 | 512 | 1.000x |
| Shared bank conflicts | 917504 | 960512 | 1.047x |
| Shared wavefronts | 1791488 | 1496064 | 0.835x |

## 1024x128 causal=0

Kernel：Br1 `<unnamed>::tiled_attention_kernel(const float *, const float *, const float *, float *, int, int, bool)`；Br4 `<unnamed>::query_tiled_kernel(const float *, const float *, const float *, float *, int, int, bool)`。

| Metric | Br1 | Br4 | Br4 / Br1 |
| --- | ---: | ---: | ---: |
| Block | (128, 1, 1) | (128, 1, 1) | - |
| Grid | (1024, 1, 1) | (256, 1, 1) | - |
| ncu duration | 1.650752 ms | 1.604064 ms | 0.972x |
| Registers/thread | 31 register/thread | 34 register/thread | 1.097x |
| Static SMEM/block | 17.484000 Kbyte/block | 20.784000 Kbyte/block | 1.189x |
| Waves/SM | 1.05 | 0.34 | 0.324x |
| SMEM limit blocks/SM | 9.000000 block | 7.000000 block | 0.778x |
| Register limit blocks/SM | 16.000000 block | 12.000000 block | 0.750x |
| Achieved occupancy | 47.636050 % | 14.815432 % | 0.311x |
| Eligible warps/cycle | 0.582620 warp | 0.152008 warp | 0.261x |
| Long scoreboard | 9.527145 inst | 7.896843 inst | 0.829x |
| Short scoreboard | 1.621581 inst | 2.432368 inst | 1.500x |
| Barrier stall | 7.032976 inst | 2.567875 inst | 0.365x |
| Warp latency | 24.355236 cycle | 17.389664 cycle | 0.714x |
| SM throughput | 24.904691 % | 13.605556 % | 0.546x |
| DRAM throughput | 0.049686 % | 0.051562 % | 1.038x |
| L2 throughput | 1.914802 % | 1.365220 % | 0.713x |
| L1 throughput | 33.185879 % | 26.241338 % | 0.791x |
| DRAM read bytes | 1.587328 Mbyte | 1.600640 Mbyte | 1.008x |
| DRAM write bytes | 0 byte | 0 byte | - |
| L2 read sectors | 5223163.000000 sector | 3602696.000000 sector | 0.690x |
| L2 write sectors | 26930.000000 sector | 29916.000000 sector | 1.111x |
| L2 sector hit rate | 96.384053 % | 97.524178 % | 1.012x |
| Global load sectors | 33570816.000000 sector | 8404992.000000 sector | 0.250x |
| Global load requests | 8392704 | 2101248 | 0.250x |
| Global store sectors | 16384.000000 sector | 16384.000000 sector | 1.000x |
| Global store requests | 4096 | 4096 | 1.000x |
| Shared bank conflicts | 29444254 | 29974528 | 1.018x |
| Shared wavefronts | 54247696 | 46233600 | 0.852x |

## 1024x128 causal=1

Kernel：Br1 `<unnamed>::tiled_attention_kernel(const float *, const float *, const float *, float *, int, int, bool)`；Br4 `<unnamed>::query_tiled_kernel(const float *, const float *, const float *, float *, int, int, bool)`。

| Metric | Br1 | Br4 | Br4 / Br1 |
| --- | ---: | ---: | ---: |
| Block | (128, 1, 1) | (128, 1, 1) | - |
| Grid | (1024, 1, 1) | (256, 1, 1) | - |
| ncu duration | 1.338144 ms | 1.318496 ms | 0.985x |
| Registers/thread | 31 register/thread | 34 register/thread | 1.097x |
| Static SMEM/block | 17.484000 Kbyte/block | 20.784000 Kbyte/block | 1.189x |
| Waves/SM | 1.05 | 0.34 | 0.324x |
| SMEM limit blocks/SM | 9.000000 block | 7.000000 block | 0.778x |
| Register limit blocks/SM | 16.000000 block | 12.000000 block | 0.750x |
| Achieved occupancy | 44.865381 % | 14.812847 % | 0.330x |
| Eligible warps/cycle | 0.748329 warp | 0.172231 warp | 0.230x |
| Long scoreboard | 8.064809 inst | 7.464040 inst | 0.926x |
| Short scoreboard | 0.749501 inst | 1.723486 inst | 2.300x |
| Barrier stall | 3.668128 inst | 1.465509 inst | 0.400x |
| Warp latency | 19.198183 cycle | 15.196087 cycle | 0.792x |
| SM throughput | 28.378063 % | 15.516628 % | 0.547x |
| DRAM throughput | 0.061295 % | 0.062730 % | 1.023x |
| L2 throughput | 2.706080 % | 1.675490 % | 0.619x |
| L1 throughput | 29.048611 % | 19.841180 % | 0.683x |
| DRAM read bytes | 1.587328 Mbyte | 1.600640 Mbyte | 1.008x |
| DRAM write bytes | 0 byte | 0 byte | - |
| L2 read sectors | 6506655.000000 sector | 3635131.000000 sector | 0.559x |
| L2 write sectors | 27089.000000 sector | 28194.000000 sector | 1.041x |
| L2 sector hit rate | 105.125244 % | 97.507348 % | 0.928x |
| Global load sectors | 33570816.000000 sector | 8404992.000000 sector | 0.250x |
| Global load requests | 8392704 | 2101248 | 0.250x |
| Global store sectors | 16384.000000 sector | 16384.000000 sector | 1.000x |
| Global store requests | 4096 | 4096 | 1.000x |
| Shared bank conflicts | 14838601 | 15068672 | 1.016x |
| Shared wavefronts | 35659218 | 27957248 | 0.784x |
