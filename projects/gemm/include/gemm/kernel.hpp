#pragma once

#include <cuda_runtime_api.h>

#include <string_view>
#include <vector>

namespace gemm {

struct Problem {
    int m;
    int n;
    int k;
};

struct LaunchResult {
    // Must point to a string literal or static/thread storage so it remains
    // valid after the launcher returns.
    const char *selected_path;
    bool used_fallback;
};

using LaunchFn = LaunchResult (*)(const float *, const float *, float *,
                                  Problem, cudaStream_t);

struct KernelDescriptor {
    const char *name;
    LaunchFn launch;
    // True for author-written optimization kernels; false for reference or
    // vendor baselines. Future reporting scripts may filter kernels using this
    // field.
    bool author_kernel;
};

LaunchResult launch_naive(const float *a, const float *b, float *c,
                          Problem problem, cudaStream_t stream);

LaunchResult launch_shared_tiled(const float *a, const float *b, float *c,
                                 Problem problem, cudaStream_t stream);

const KernelDescriptor *find_kernel(std::string_view name);
std::vector<KernelDescriptor> registered_kernels();

} // namespace gemm
