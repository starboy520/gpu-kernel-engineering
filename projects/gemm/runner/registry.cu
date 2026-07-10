#include "gemm/kernel.hpp"

namespace gemm {

const KernelDescriptor* find_kernel(std::string_view) {
    return nullptr;
}

std::vector<KernelDescriptor> registered_kernels() {
    return {};
}

}  // namespace gemm
