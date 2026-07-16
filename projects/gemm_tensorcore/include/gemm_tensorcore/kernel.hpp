#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace gemm_tensorcore {

inline constexpr int wmma_m = 16;
inline constexpr int wmma_n = 16;
inline constexpr int wmma_k = 16;

// G1 contract: A/B are row-major FP16 16x16 matrices; C is row-major FP32
// 16x16. One Warp computes one complete tile. The launch is asynchronous.
void launch_wmma_single(const __half *a, const __half *b, float *c,
                        cudaStream_t stream);

} // namespace gemm_tensorcore
