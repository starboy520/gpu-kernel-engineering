#include "flash_attention/kernel.hpp"

#include <array>
#include <limits>
#include <stdexcept>

namespace {

std::size_t naive_workspace_bytes(flash_attention::Problem problem) {
    const std::size_t n = static_cast<std::size_t>(problem.n);
    if (n > std::numeric_limits<std::size_t>::max() / n ||
        n * n > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("naive scores workspace size overflow");
    }
    return n * n * sizeof(float);
}

std::size_t no_workspace_bytes(flash_attention::Problem) { return 0; }

const std::array<flash_attention::KernelDescriptor, 4> kernel_table{{
    {"naive", flash_attention::launch_naive_materialized, naive_workspace_bytes,
     true},
    {"tiled", flash_attention::launch_tiled_online, no_workspace_bytes, true},
    {"tiled-parallel", flash_attention::launch_tiled_parallel,
     no_workspace_bytes, true},
    {"tiled-async", flash_attention::launch_tiled_async, no_workspace_bytes,
     true},
}};

} // namespace

namespace flash_attention {

const KernelDescriptor *find_kernel(std::string_view name) {
    for (const KernelDescriptor &kernel : kernel_table) {
        if (name == kernel.name) {
            return &kernel;
        }
    }
    return nullptr;
}

std::vector<KernelDescriptor> registered_kernels() {
    return {kernel_table.begin(), kernel_table.end()};
}

} // namespace flash_attention