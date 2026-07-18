#include "gemm_tensorcore/kernel.hpp"
#include "gpu_kernel/cuda_check.hpp"

#include <mma.h>
#include <stdexcept>

namespace {

constexpr int kThreadsPerBlock = 32;
namespace wmma = nvcuda::wmma;

// 这里已经限定m, n, k 是  16的倍数；
__global__ void wmma_direct_kernel(const __half *a, const __half *b, float *c,
                                   int m, int n, int k) {

    // Address invariants to derive in the implementation:
    //   m0 = blockIdx.y * 16
    //   n0 = blockIdx.x * 16
    //   注意， B col-major， 所以 B tile start计算要注意,
    //   C tile starts at C[m0, n0] with ldc=n

    // 声明a, b c,
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> m_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> m_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // 置零
    wmma::fill_fragment(acc_frag, 0.0f);

    int m0 = blockIdx.y * 16;
    int n0 = blockIdx.x * 16;

    for (int kk = 0; kk < k; kk += 16) {
        // load a
        wmma::load_matrix_sync(m_a, a + m0 * k + kk, k);
        // b  col-major,  这里需要注意， 其实就是n*k 的  row-major
        wmma::load_matrix_sync(m_b, b + n0 * k + kk, k);

        wmma::mma_sync(acc_frag, m_a, m_b, acc_frag);
    }

    // 写回
    wmma::store_matrix_sync(c + m0 * n + n0, acc_frag, n, wmma::mem_row_major);
}

} // namespace

void gemm_tensorcore::launch_wmma_direct(const __half *a, const __half *b,
                                         float *c, Problem padded_problem,
                                         cudaStream_t stream) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument(
            "Direct WMMA launch requires non-null A/B/C buffers");
    }
    if (padded_problem.m <= 0 || padded_problem.n <= 0 ||
        padded_problem.k <= 0 || padded_problem.m % wmma_m != 0 ||
        padded_problem.n % wmma_n != 0 || padded_problem.k % wmma_k != 0) {
        throw std::invalid_argument(
            "Direct WMMA launch requires positive dimensions divisible by 16");
    }

    const dim3 block(kThreadsPerBlock);
    const dim3 grid(padded_problem.n / wmma_n, padded_problem.m / wmma_m);
    wmma_direct_kernel<<<grid, block, 0, stream>>>(
        a, b, c, padded_problem.m, padded_problem.n, padded_problem.k);
    GPU_CUDA_CHECK(cudaGetLastError());
}