#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace gemm {

inline const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

[[noreturn]] inline void throw_cuda_error(cudaError_t status, const char* expression,
                                          const char* file, int line) {
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << " with "
            << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

[[noreturn]] inline void throw_cublas_error(cublasStatus_t status,
                                            const char* expression,
                                            const char* file, int line) {
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << " with "
            << cublas_status_name(status);
    throw std::runtime_error(message.str());
}

}  // namespace gemm

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        const cudaError_t cuda_check_status__ = (expr);                       \
        if (cuda_check_status__ != cudaSuccess) {                             \
            ::gemm::throw_cuda_error(cuda_check_status__, #expr, __FILE__,    \
                                     __LINE__);                               \
        }                                                                     \
    } while (false)

#define CUBLAS_CHECK(expr)                                                    \
    do {                                                                      \
        const cublasStatus_t cublas_check_status__ = (expr);                  \
        if (cublas_check_status__ != CUBLAS_STATUS_SUCCESS) {                 \
            ::gemm::throw_cublas_error(cublas_check_status__, #expr, __FILE__,\
                                       __LINE__);                             \
        }                                                                     \
    } while (false)
