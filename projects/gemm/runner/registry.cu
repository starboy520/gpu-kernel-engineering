#include "gemm/kernel.hpp"

#include <array>

namespace {

const std::array<gemm::KernelDescriptor, 6> kernel_table{{
    {"naive", gemm::launch_naive, true},
    {"shared", gemm::launch_shared_tiled, true},
    {"register", gemm::launch_register_tiled, true},
    {"vectorized", gemm::launch_vectorized_tiled, true},
    {"async-16b", gemm::launch_double_buffer, true},
    {"cublas-fp32", gemm::launch_cublas_fp32, false},
}};

} // namespace

namespace gemm {

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

} // namespace gemm
