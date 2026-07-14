#pragma once

#include <cuda_runtime_api.h>

namespace attention_prefill {

inline constexpr int max_head_dimension = 128;

struct Problem {
    int n;
    int d;
    bool causal;
};

// Q/K/V/output are non-null, non-overlapping dense row-major buffers with n*d
// FP32 elements. Requires n>0, 1<=d<=max_head_dimension, n*d<=INT_MAX, and a
// valid CUDA stream. The launch is asynchronous with respect to the host.
void launch_query_tiled(const float *q, const float *k, const float *v,
                        float *output, Problem problem, cudaStream_t stream);

} // namespace attention_prefill
