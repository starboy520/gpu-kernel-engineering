#include "gemm_tensorcore/kernel.hpp"
#include "gpu_kernel/cuda_check.hpp"

#include <mma.h>
#include <stdexcept>

namespace {

constexpr int kThreadsPerBlock = 32;

__global__ void wmma_single_kernel(const __half *a, const __half *b, float *c) {
    // TODO(G1): 学习者实现一个 Warp 的 WMMA 16x16x16。
    //
    // 固定合同：
    //   A：matrix_a fragment，FP16，row-major
    //   B：matrix_b fragment，FP16，row-major
    //   accumulator：FP32
    //   C：FP32 row-major
    //
    // 推荐实现顺序：
    //   1. 声明 A/B/accumulator fragment；
    //   2. accumulator 填 0；
    //   3. 整个 Warp 一致执行 load_matrix_sync；
    //   4. 整个 Warp 一致执行 mma_sync；
    //   5. 整个 Warp 一致执行 store_matrix_sync。
    //
    // 不要只让 lane 0 调用 WMMA；G1 不处理 tail、循环或性能优化。
    (void)a;
    (void)b;
    (void)c;
}

} // namespace

void gemm_tensorcore::launch_wmma_single(const __half *a, const __half *b,
                                         float *c, cudaStream_t stream) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument(
            "WMMA single-tile launch requires non-null A/B/C buffers");
    }

    wmma_single_kernel<<<1, kThreadsPerBlock, 0, stream>>>(a, b, c);
    GPU_CUDA_CHECK(cudaGetLastError());
}
