#include "gemm/cuda_check.hpp"
#include "gemm/kernel.hpp"
#include <cstdint>

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

// CUDA 13.3 diagnoses the documented shared pipeline state declaration as
// dynamic initialization. make_pipeline() initializes its barriers and
// reference count explicitly, so suppress only that diagnostic in this file.
#pragma nv_diag_suppress 20054

// pipeline 使用
//  cooperative_group:
//__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss;
// auto block  = cg::this_thread_block();
// auto pipe = cuda::make_pipeline(block, &pss);

// memcpy_async
//  memcpy_async(_Type* __destination, _Type const* __source, _Size __size,
//  pipeline<_Scope>& __pipeline)

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 4;

__device__ void load_async_16b(float *s_mem, int smem_ld, const float *src,
                               int tileH, int tileW, int row_base, int col_base,
                               int ld, int bound_row, int bound_col, int tid,
                               int n_thread,
                               cuda::pipeline<cuda::thread_scope_block> &pipe) {
    for (int i = tid; i < tileW * tileH / 4; i += n_thread) {
        int tile_row = i * 4 / tileW;
        int tile_col = i * 4 % tileW;
        int g_row = row_base + tile_row;
        int g_col = col_base + tile_col;
        if (g_row < bound_row && g_col + 3 < bound_col) {
            cuda::memcpy_async(
                &s_mem[tile_row * smem_ld + tile_col],
                &src[g_row * ld + g_col],
                cuda::aligned_size_t<16>(sizeof(float4)), pipe);
        } else {
            for (int offset = 0; offset < 4; ++offset) {
                s_mem[tile_row * smem_ld + tile_col + offset] = 0.0f;
            }
        }
    }
}

__global__ void double_buffer_kernel(const float *a, const float *b, float *c,
                                     int m, int n, int k) {

    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    static_assert((BM / TM) * (BN / TN) <= 1024);

    __shared__ __align__(16) float s_a[2][BM][BK];
    __shared__ __align__(16) float s_b[2][BK][BN];

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss;
    auto block = cooperative_groups::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pss);

    // block内部， thread 打平
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y; // 一个block多少线程

    int global_start_row = BM * blockIdx.y;
    int global_start_col = BN * blockIdx.x;

    // 累加器, 一个线程负责TM*TN;
    float reg_acc[TM][TN] = {0.0f};

    int stage = 0;
    // 第一步预取
    int step = 0;
    pipe.producer_acquire();
    load_async_16b(&s_a[stage][0][0], BK, a, BM, BK, global_start_row, step, k,
                   m, k, tid, stride, pipe);
    load_async_16b(&s_b[stage][0][0], BN, b, BK, BN, step, global_start_col, n,
                   k, n, tid, stride, pipe);
    pipe.producer_commit();

    // 整体还是沿着k方向
    for (; step < k; step += BK) {
        //  协助搬运，  打平thread.
        int next_stage = stage == 0 ? 1 : 0;

        if (step + BK < k) {
            pipe.producer_acquire();
            load_async_16b(&s_a[next_stage][0][0], BK, a, BM, BK,
                           global_start_row, step + BK, k, m, k, tid, stride,
                           pipe);
            load_async_16b(&s_b[next_stage][0][0], BN, b, BK, BN, step + BK,
                           global_start_col, n, k, n, tid, stride, pipe);
            pipe.producer_commit();
        }

        pipe.consumer_wait();
        block.sync(); // Make the completed async copies visible to the block.
        for (int kk = 0; kk < BK; kk++) {
            float reg_t_a[TM];
            float reg_t_b[TN];
            for (int j = 0; j < TM; j++) {
                reg_t_a[j] = s_a[stage][threadIdx.y * TM + j][kk];
            }
            for (int j = 0; j < TN; j++) {
                reg_t_b[j] = s_b[stage][kk][threadIdx.x * TN + j];
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    reg_acc[i][j] += reg_t_a[i] * reg_t_b[j];
                }
            }
        }
        block.sync();
        pipe.consumer_release();
        stage = next_stage;
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

gemm::LaunchResult gemm::launch_double_buffer(const float *a, const float *b,
                                              float *c, gemm::Problem problem,
                                              cudaStream_t stream) {
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);
    dim3 block(BN / TN, BM / TM);
    dim3 grid((BN + problem.n - 1) / BN, (BM + problem.m - 1) / BM);

    const bool aligned_a =
        reinterpret_cast<std::uintptr_t>(a) % (alignof(float4)) == 0;
    const bool aligned_b =
        reinterpret_cast<std::uintptr_t>(b) % (alignof(float4)) == 0;

    const bool can_use_async =
        aligned_a && aligned_b && problem.k % 4 == 0 && problem.n % 4 == 0;

    if (!can_use_async) {
        gemm::launch_register_tiled(a, b, c, problem, stream);
        return {"fallback-register", true};
    } else {
        double_buffer_kernel<<<grid, block, 0, stream>>>(a, b, c, problem.m,
                                                         problem.n, problem.k);
        CUDA_CHECK(cudaGetLastError());

        return {"fast-pipeline-16b", false};
    }
}