#include "flash_attention/cuda_check.hpp"
#include "flash_attention/kernel.hpp"

#include <cstdint>

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#pragma nv_diag_suppress 20054
namespace {

constexpr int BC = 16;
constexpr int MAX_D = 128;
constexpr int PIPELINE_STAGES = 2;

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
            cuda::memcpy_async(&s_mem[tile_row * smem_ld + tile_col],
                               &src[g_row * ld + g_col],
                               cuda::aligned_size_t<16>(sizeof(float4)), pipe);
        } else {
            for (int offset = 0; offset < 4; ++offset) {
                s_mem[tile_row * smem_ld + tile_col + offset] = 0.0f;
            }
        }
    }
}

__global__ void tiled_async_attention_kernel(const float *q, const float *k,
                                             const float *v, float *output,
                                             int n, int d, bool causal) {
    __shared__ float s_q[MAX_D];
    __shared__ __align__(16) float s_k[PIPELINE_STAGES][BC][MAX_D];
    __shared__ __align__(16) float s_v[PIPELINE_STAGES][BC][MAX_D];
    __shared__ float m;
    __shared__ float l;
    __shared__ float score[BC];
    __shared__ float alpha;
    __shared__ float acc[MAX_D];

    int stage = 0;
    float scale = rsqrt((float)d);

    __shared__
        cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_STAGES>
            pss;
    auto block = cooperative_groups::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pss);

    if (threadIdx.x == 0) {
        m = -INFINITY;
        l = 0.0f;
        alpha = 0.0f;
    }
    __syncthreads();

    // 一个block 负责一行
    int query = blockIdx.x;
    const float *q_row = q + query * d;
    float *o_row = output + query * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        s_q[i] = q_row[i];
        acc[i] = 0.0f;
    }

    __syncthreads();

    int stride = blockDim.x;
    int start = 0;
    int tid = threadIdx.x;

    int valid = min(BC, n - start);
    pipe.producer_acquire();
    load_async_16b(&s_k[stage][0][0], MAX_D, k, valid, d, start, 0, d, n, d,
                   tid, stride, pipe);
    load_async_16b(&s_v[stage][0][0], MAX_D, v, valid, d, start, 0, d, n, d,
                   tid, stride, pipe);
    pipe.producer_commit();
    while (start < n) {
        int next_stage = (stage + 1) % 2;
        int next_valid = min(BC, n - (start + valid));
        if (next_valid > 0) {
            pipe.producer_acquire();
            load_async_16b(&s_k[next_stage][0][0], MAX_D, k, next_valid, d,
                           start + valid, 0, d, n, d, tid, stride, pipe);
            load_async_16b(&s_v[next_stage][0][0], MAX_D, v, next_valid, d,
                           start + valid, 0, d, n, d, tid, stride, pipe);
            pipe.producer_commit();
        }

        pipe.consumer_wait();
        block.sync();
        if (threadIdx.x < valid) {
            // 每个thread 算一个 query
            // TODO casual mask
            int cur_key = threadIdx.x;
            if (cur_key + start > query && causal) {
                // do somthing
                score[cur_key] = -INFINITY;
            } else {
                float sum = 0.0f;
                for (int i = 0; i < d; i++) {
                    sum += s_q[i] * s_k[stage][cur_key][i];
                }

                score[cur_key] = sum * scale;
            }
        }
        block.sync();

        if (threadIdx.x == 0) {
            float cur_max = -INFINITY;
            for (int i = 0; i < valid; i++) {
                cur_max = fmax(cur_max, score[i]);
            }

            if (isinf(cur_max) && cur_max < 0) {
                alpha = 1.0f;
                for (int i = 0; i < valid; i++)
                    score[i] = 0.0f;
            } else {
                float m_new = fmax(m, cur_max);
                alpha = isinf(m) ? 0 : expf(m - m_new);
                float tile_l = 0;
                for (int i = 0; i < valid; i++) {
                    float w = expf(score[i] - m_new);
                    score[i] = w;
                    tile_l += w;
                }
                l = l * alpha + tile_l;
                m = m_new;
            }
        }
        block.sync();

        for (int feature = threadIdx.x; feature < d; feature += blockDim.x) {
            float add = 0.0;
            for (int i = 0; i < valid; i++) {
                add += score[i] * s_v[stage][i][feature];
            }
            acc[feature] = acc[feature] * alpha + add;
        }
        pipe.consumer_release();
        block.sync();

        start += valid;
        stage = next_stage;
        valid = next_valid;
    }

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        o_row[i] = acc[i] / l;
    }
}

} // namespace

flash_attention::LaunchResult flash_attention::launch_tiled_async(
    const float *q, const float *k, const float *v, float *output,
    float *workspace, Problem problem, cudaStream_t stream) {
    (void)workspace;

    const bool aligned_k =
        reinterpret_cast<std::uintptr_t>(k) % alignof(float4) == 0;
    const bool aligned_v =
        reinterpret_cast<std::uintptr_t>(v) % alignof(float4) == 0;
    const bool can_use_async = aligned_k && aligned_v && problem.d % 4 == 0;

    if (!can_use_async) {
        launch_tiled_online(q, k, v, output, workspace, problem, stream);
        return {"fallback-tiled", true};
    }

    tiled_async_attention_kernel<<<problem.n, 128, 0, stream>>>(
        q, k, v, output, problem.n, problem.d, problem.causal);
    FA_CUDA_CHECK(cudaGetLastError());
    return {"fast-pipeline-16b", false};
}