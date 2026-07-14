#include "gemm/kernel.hpp"
#include "gpu_kernel/cuda_check.hpp"

#include <cstdint>

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 4;

__global__ void vectorized_tiled_kernel(const float *a, const float *b,
                                        float *c, int m, int n, int k) {

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
        for (int i = tid; i < BK * BM / 4; i += stride) {
            int tile_row = i * 4 / BK;
            int tile_col = i * 4 % BK;

            int g_row = global_start_row + tile_row;
            int g_col = step + tile_col;
            if (g_row < m && g_col + 3 < k) {
                const float4 v_a =
                    reinterpret_cast<const float4 *>(a + g_row * k + g_col)[0];
                s_a[tile_row][tile_col] = v_a.x;
                s_a[tile_row][tile_col + 1] = v_a.y;
                s_a[tile_row][tile_col + 2] = v_a.z;
                s_a[tile_row][tile_col + 3] = v_a.w;
            } else {
                for (int offset = 0; offset < 4; offset++) {
                    s_a[tile_row][tile_col + offset] = 0.0f;
                }
            }
        }

        for (int i = tid; i < BN * BK / 4; i += stride) {
            int tile_row = i * 4 / BN;
            int tile_col = i * 4 % BN;
            int g_row = step + tile_row;
            int g_col = global_start_col + tile_col;
            if (g_row < k && g_col + 3 < n) {
                const float4 v_b =
                    reinterpret_cast<const float4 *>(b + g_row * n + g_col)[0];
                s_b[tile_row][tile_col] = v_b.x;
                s_b[tile_row][tile_col + 1] = v_b.y;
                s_b[tile_row][tile_col + 2] = v_b.z;
                s_b[tile_row][tile_col + 3] = v_b.w;
            } else {
                for (int offset = 0; offset < 4; offset++) {
                    s_b[tile_row][tile_col + offset] = 0.0f;
                }
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

gemm::LaunchResult gemm::launch_vectorized_tiled(const float *a, const float *b,
                                                 float *c,
                                                 gemm::Problem problem,
                                                 cudaStream_t stream) {
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);
    dim3 block(BN / TN, BM / TM);
    dim3 grid((BN + problem.n - 1) / BN, (BM + problem.m - 1) / BM);

    const bool aligned_a =
        reinterpret_cast<std::uintptr_t>(a) % (alignof(float4)) == 0;
    const bool aligned_b =
        reinterpret_cast<std::uintptr_t>(b) % (alignof(float4)) == 0;

    const bool can_use_float4 =
        aligned_a && aligned_b && problem.k % 4 == 0 && problem.n % 4 == 0;

    if (!can_use_float4) {
        gemm::launch_register_tiled(a, b, c, problem, stream);
        return {"fallback-register", true};
    } else {
        vectorized_tiled_kernel<<<grid, block, 0, stream>>>(
            a, b, c, problem.m, problem.n, problem.k);
        GPU_CUDA_CHECK(cudaGetLastError());

        return {"fast-float4", false};
    }
}