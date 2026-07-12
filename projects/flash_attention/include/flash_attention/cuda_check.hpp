#pragma once

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace flash_attention {

[[noreturn]] inline void throw_cuda_error(cudaError_t status,
                                          const char *expression,
                                          const char *file, int line) {
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << " with "
            << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

} // namespace flash_attention

#define FA_CUDA_CHECK(expr)                                                    \
    do {                                                                       \
        const cudaError_t fa_cuda_status__ = (expr);                           \
        if (fa_cuda_status__ != cudaSuccess) {                                 \
            ::flash_attention::throw_cuda_error(fa_cuda_status__, #expr,       \
                                                __FILE__, __LINE__);           \
        }                                                                      \
    } while (false)
