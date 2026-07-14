#include "gemm/reference.hpp"

#include "gemm/cublas_check.hpp"
#include "gemm/kernel.hpp"

#include <cublas_v2.h>

namespace gemm {
namespace {

class CublasHandle {
  public:
    CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }

    ~CublasHandle() { (void)cublasDestroy(handle_); }

    CublasHandle(const CublasHandle &) = delete;
    CublasHandle &operator=(const CublasHandle &) = delete;

    cublasHandle_t get() const { return handle_; }

  private:
    cublasHandle_t handle_{};
};

cublasHandle_t persistent_handle() {
    thread_local CublasHandle handle;
    return handle.get();
}

} // namespace

void reference_cublas_device(const float *device_a, const float *device_b,
                             float *device_c, Problem problem,
                             cudaStream_t stream) {
    const cublasHandle_t handle = persistent_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, problem.n,
                             problem.m, problem.k, &alpha, device_b, problem.n,
                             device_a, problem.k, &beta, device_c, problem.n));
}

LaunchResult launch_cublas_fp32(const float *a, const float *b, float *c,
                                Problem problem, cudaStream_t stream) {
    reference_cublas_device(a, b, c, problem, stream);
    return {"cublas-pedantic-fp32", false};
}

} // namespace gemm