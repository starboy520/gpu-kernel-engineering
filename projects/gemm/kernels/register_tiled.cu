#include "gemm/cuda_check.hpp"
#include "gemm/kernel.hpp"

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 4;

__global__ void register_tiled_kernel(const float *a, const float *b, float *c,
                                      int m, int n, int k) {

    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    static_assert((BM / TM) * (BN / TN) <= 1024);

    __shared__ float s_a[BM][BK + 1];
    __shared__ float s_b[BK][BN + 1];

    // block内部， thread 打平
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y; // 一个block多少线程

    int global_start_row = BM * blockIdx.y;
    int global_start_col = BN * blockIdx.x;

    // 累加器, 一个线程负责TM*TN;
    float reg_acc[TM][TN] = {0.0f};

    // 整体还是沿着k方向
    for (int step = 0; step < k; step += BK) {
        //  协助搬运，  打平thread.
        for (int i = tid; i < BK * BM; i += stride) {
            int tile_row = i / BK;
            int tile_col = i % BK;

            int g_row = global_start_row + tile_row;
            int g_col = step + tile_col;
            if (g_row < m && g_col < k) {
                s_a[tile_row][tile_col] = a[g_row * k + g_col];
            } else {
                s_a[tile_row][tile_col] = 0.0f;
            }
        }

        for (int i = tid; i < BN * BK; i += stride) {
            int tile_row = i / BN;
            int tile_col = i % BN;
            int g_row = step + tile_row;
            int g_col = global_start_col + tile_col;
            if (g_row < k && g_col < n) {
                s_b[tile_row][tile_col] = b[g_row * n + g_col];
            } else {
                s_b[tile_row][tile_col] = 0.0f;
            }
        }
        __syncthreads();

        for (int kk = 0; kk < BK; kk++) {
            float reg_t_a[TM];
            float reg_t_b[TN];
            for (int j = 0; j < TM; j++) {
                reg_t_a[j] = s_a[threadIdx.y * TM + j][kk];
            }
            for (int j = 0; j < TN; j++) {
                reg_t_b[j] = s_b[kk][threadIdx.x * TN + j];
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    reg_acc[i][j] += reg_t_a[i] * reg_t_b[j];
                }
            }
        }
        __syncthreads();
    }
    // 累加器写回
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            // global_start_row + threadidx.y*tm + i,
            // global_start_col+threadidx.x*tn+j
            int g_row = global_start_row + threadIdx.y * TM + i;
            int g_col = global_start_col + threadIdx.x * TN + j;
            if (g_row < m && g_col < n) {
                c[g_row * n + g_col] = reg_acc[i][j];
            }
        }
    }
}

gemm::LaunchResult gemm::launch_register_tiled(const float *a, const float *b,
                                               float *c, gemm::Problem problem,
                                               cudaStream_t stream) {
    dim3 block(BN / TN, BM / TM);
    dim3 grid((BN + problem.n - 1) / BN, (BM + problem.m - 1) / BM);
    register_tiled_kernel<<<grid, block, 0, stream>>>(a, b, c, problem.m,
                                                      problem.n, problem.k);
    CUDA_CHECK(cudaGetLastError());

    return {"register", false};
}