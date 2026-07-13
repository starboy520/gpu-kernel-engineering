#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string_view>
#include <vector>

namespace flash_attention {

struct Problem {
    int n;
    int d;
    bool causal;
};

struct LaunchResult {
    // Must point to a string literal or static/thread storage so it remains
    // valid after the launcher returns.
    const char *selected_path;
    bool used_fallback;
};

using LaunchFn = LaunchResult (*)(const float *q, const float *k,
                                  const float *v, float *output,
                                  float *workspace, Problem problem,
                                  cudaStream_t stream);
using WorkspaceBytesFn = std::size_t (*)(Problem problem);

struct KernelDescriptor {
    const char *name;
    LaunchFn launch;
    WorkspaceBytesFn workspace_bytes;
    // True for author-written kernels; false for future vendor baselines.
    bool author_kernel;
};

LaunchResult launch_naive_materialized(const float *q, const float *k,
                                       const float *v, float *output,
                                       float *scores, Problem problem,
                                       cudaStream_t stream);

LaunchResult launch_tiled_online(const float *q, const float *k, const float *v,
                                 float *output, float *workspace,
                                 Problem problem, cudaStream_t stream);

const KernelDescriptor *find_kernel(std::string_view name);
std::vector<KernelDescriptor> registered_kernels();

} // namespace flash_attention