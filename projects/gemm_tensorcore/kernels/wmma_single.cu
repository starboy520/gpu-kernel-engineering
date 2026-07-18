#include "gemm_tensorcore/kernel.hpp"
#include "gpu_kernel/cuda_check.hpp"

#include <mma.h>
#include <stdexcept>

namespace {

constexpr int kThreadsPerBlock = 32;
namespace wmma = nvcuda::wmma;

// 这个kernel 只是实现一个warp 内部，32个线程， 怎么协作加载， 算c= a* b
// 这里只有一个很简单的假设， a 16 * 16, b 16* 16, c 16* 16, 所以一次算就行
// 这里row_major,

__global__ void wmma_single_kernel(const __half *a, const __half *b, float *c) {
    // 声明a, b c,
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> m_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> m_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // 置零
    wmma::fill_fragment(acc_frag, 0.0f);

    // 协作加载a,b
    wmma::load_matrix_sync(m_a, a, 16);
    wmma::load_matrix_sync(m_b, b, 16);
    // 计算
    wmma::mma_sync(acc_frag, m_a, m_b, acc_frag);

    // 写回
    wmma::store_matrix_sync(c, acc_frag, 16, wmma::mem_row_major);
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
