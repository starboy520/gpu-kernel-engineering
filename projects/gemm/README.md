# FP32 GEMM Optimization Ladder

**Status:** Active rebuild

## Scope

Row-major $C = A \times B$ in FP32 on NVIDIA A100 (`sm_80`).

## Optimization Ladder

1. Naive
2. Shared-memory tiling
3. 2D register tiling
4. `float4` vectorized load
5. `cp.async` double buffering
6. cuBLAS pedantic FP32 baseline

Results will be reported only from reproducible measurements with correctness checks.