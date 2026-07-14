#pragma once

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace gpu_kernel {

[[noreturn]] inline void throw_cuda_error(cudaError_t status,
                                          const char *expression,
                                          const char *file, int line) {
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << " with "
            << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

} // namespace gpu_kernel

#define GPU_CUDA_CHECK(expr)                                                   \
    do {                                                                       \
        const cudaError_t gpu_cuda_status_value = (expr);                      \
        if (gpu_cuda_status_value != cudaSuccess) {                            \
            ::gpu_kernel::throw_cuda_error(gpu_cuda_status_value, #expr,       \
                                           __FILE__, __LINE__);                \
        }                                                                      \
    } while (false)
