#pragma once

#include "attention_prefill/query_tiled.hpp"

#include <cuda_runtime_api.h>

namespace attention_prefill {

inline constexpr int m2_queries_per_cta = 4;

// Full M2 contract: four warps per CTA, one warp owns one Query row.
// Q/K/V/output follow the same dense FP32 row-major contract as M1.
// The launch is asynchronous; the caller owns synchronization.
void launch_warp_per_query(const float *q, const float *k, const float *v,
                           float *output, Problem problem, cudaStream_t stream);

} // namespace attention_prefill
