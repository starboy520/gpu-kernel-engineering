#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace gemm_tensorcore {

inline constexpr int wmma_m = 16;
inline constexpr int wmma_n = 16;
inline constexpr int wmma_k = 16;

struct Problem {
    int m;
    int n;
    int k;
};

// G1 contract: A/B are row-major FP16 16x16 matrices; C is row-major FP32
// 16x16. One Warp computes one complete tile. The launch is asynchronous.
void launch_wmma_single(const __half *a, const __half *b, float *c,
                        cudaStream_t stream);

// G1.5 Direct contract: dimensions are positive multiples of 16. A is
// row-major FP16, B is col-major FP16, and C is row-major FP32. One Warp/block
// computes one 16x16 C tile. The launch remains asynchronous.
void launch_wmma_direct(const __half *a, const __half *b, float *c,
                        Problem padded_problem, cudaStream_t stream);

} // namespace gemm_tensorcore
