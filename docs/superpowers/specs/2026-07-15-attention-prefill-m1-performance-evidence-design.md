# Attention Prefill M1 Performance Evidence Design

## Goal

Build a reproducible, strict single-variable comparison between the frozen `Br=1,Bc=16` FP32 SIMT Attention baseline and the M1 `Br=4,Bc=16` FP32 SIMT implementation on A100 `sm_80`.

## Comparison Contract

The two paths use identical FP32 Q/K/V inputs, seed, causal mode, `Bc=16`, 128-thread CTA, CPU-double reference, tolerance, CUDA Event timing code, warmup, iterations, repeats, compiler flags, GPU, and output schema. The only intended algorithmic variable is Query rows per CTA: `Br=1` versus `Br=4`.

A dedicated evidence runner links both existing kernel libraries and owns validation and timing. Neither kernel body is changed.

## Evidence Layers

1. Canonical wall-clock benchmark: `N={128,256,512,1024,2048}`, `D={64,128}`, causal `{0,1}`, both implementations; warmup 10, iterations 50, repeats 3; median/min/max/spread from CUDA Event kernel-only timing.
2. Theory ledger: report CTA counts and requested K/V FP32 elements for each implementation; distinguish source-level requests from measured L2/DRAM traffic.
3. Nsight Compute: profile `256x64 causal=0`, `1024x128 causal=0`, and `1024x128 causal=1` for both paths. Capture resource, occupancy, eligible-warp, long/short-scoreboard, barrier, cache, DRAM, and shared-memory evidence. ncu duration is not canonical wall-clock evidence.
4. SASS: extract both kernel symbols from the same evidence binary and count `FFMA`, barriers, global/shared accesses, local spills, `HMMA`, and `LDGSTS`. Static counts are not runtime execution counts.

## Artifact Policy

Commit the runner, scripts, tests, reviewed canonical CSV, generated summary, compact ncu summary, compact SASS evidence, and reproduction instructions. Ignore `.ncu-rep`, complete SASS, and temporary outputs.

## Success Criteria

Both paths pass the same reference check; canonical output has 40 rows and complete provenance; ncu reports six intended kernel launches; SASS confirms FP32 SIMT without Tensor Core or async-copy instructions; the final report explains benefits, regressions, and inconclusive results without hiding negative evidence.
