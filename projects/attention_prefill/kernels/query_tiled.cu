#include "attention_prefill/query_tiled.hpp"
#include "gpu_kernel/cuda_check.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

constexpr int BR = 4;      // 每个block 处理4个query
constexpr int BC = 16;     // 每个BLOCK处理 TILE key/value 为 16个，
constexpr int MAX_D = 128; // 每个 query、key、value 最多 128 维。

__device__ void load_data(float *s_mem, int ld, int tileH, int tileW,
                          const float *src, int row_base, int col_base,
                          int ld_src, int tid, int stride) {
    for (int i = tid; i < tileH * tileW; i += stride) {
        int row = i / tileW;
        int column = i % tileW;
        int g_row = row_base + row;
        int g_col = col_base + column;
        s_mem[row * ld + column] = src[g_row * ld_src + g_col];
    }
}

/**
 * 整体流程，
 *  阶段 A：128 个线程合作加载 4 条 Q
    阶段 B：128 个线程合作加载 16 条 K/V
    阶段 C：前 64 个线程计算 4×16=64 个 score
    阶段 D：前 4 个线程分别管理 4 行 softmax 状态
    阶段 E：128 个线程合作更新 4×D 个 O_acc
 */
__global__ void query_tiled_kernel(const float *q, const float *k,
                                   const float *v, float *output, int n, int d,
                                   bool causal) {
    __shared__ float q_s[BR][MAX_D];
    __shared__ float k_s[BC][MAX_D];
    __shared__ float v_s[BC][MAX_D];
    __shared__ float scores[BR][BC];
    __shared__ float m[BR], l[BR], alpha[BR];
    __shared__ float o_acc[BR][MAX_D];

    // 每条 Query 独立初始化 Online Softmax 状态和输出累加器。
    if (threadIdx.x < BR) {
        m[threadIdx.x] = -INFINITY;
        l[threadIdx.x] = 0.0f;
        for (int i = 0; i < MAX_D; i++) {
            o_acc[threadIdx.x][i] = 0.0f;
        }
    }

    __syncthreads();

    float scale = rsqrtf(d);

    int stride = blockDim.x;
    int tid = threadIdx.x;
    int query_start = blockIdx.x * BR;
    // 防御性检查。
    if (query_start >= n) {
        return;
    }
    // A: load query;
    int valid_query = min(BR, n - query_start);
    load_data(&q_s[0][0], MAX_D, min(BR, n - query_start), d, q, query_start, 0,
              d, tid, stride);
    __syncthreads();

    for (int start_row = 0; start_row < n; start_row += BC) {
        // B: load k&v
        int valid_kv = min(BC, n - start_row);
        load_data(&k_s[0][0], MAX_D, valid_kv, d, k, start_row, 0, d, tid,
                  stride);
        load_data(&v_s[0][0], MAX_D, valid_kv, d, v, start_row, 0, d, tid,
                  stride);
        __syncthreads();

        // C: calculate the scores BR * BC;
        if (tid < valid_query * valid_kv) {

            int idx_query = tid / valid_kv;
            int idx_key = tid % valid_kv;

            // !!! 这边要特别注意， 要是全局的idx
            if (start_row + idx_key > query_start + idx_query && causal) {
                scores[idx_query][idx_key] = -INFINITY;

            } else {
                float sum = 0.0f;
                for (int i = 0; i < d; i++) {
                    sum += q_s[idx_query][i] * k_s[idx_key][i];
                }
                scores[idx_query][idx_key] = sum * scale;
            }
        }
        __syncthreads();

        // 阶段 D：前 4 个线程分别管理 4 行 softmax 状态
        if (tid < valid_query) {
            float tile_m = -INFINITY;
            for (int i = 0; i < valid_kv; i++) {
                tile_m = fmaxf(tile_m, scores[tid][i]);
            }

            if (isinf(tile_m) && tile_m < 0) {
                // 整个 K/V tile 都被 mask：保持 m/l/O_acc 不变，权重置 0。
                alpha[tid] = 1.0f;
                for (int i = 0; i < valid_kv; i++) {
                    scores[tid][i] = 0.0f;
                }
            } else {
                float m_new = fmax(tile_m, m[tid]);
                float tile_l = 0.0f;
                alpha[tid] = isinf(m[tid]) ? 0 : expf(m[tid] - m_new);
                for (int i = 0; i < valid_kv; i++) {
                    float w = expf(scores[tid][i] - m_new);
                    scores[tid][i] = w;
                    tile_l += w;
                }
                m[tid] = m_new;
                l[tid] = l[tid] * alpha[tid] + tile_l;
            }
        }
        __syncthreads();

        //  阶段 E：128 个线程合作更新 4×D 个 O_acc
        // 每个 Query 有 d 个 feature。
        for (int feature = tid; feature < valid_query * d; feature += stride) {
            int query = feature / d;
            int cur_feature = feature % d;
            // acc[query][cur_feature];
            float sum = 0.0f;
            for (int i = 0; i < valid_kv; i++) {
                sum += scores[query][i] * v_s[i][cur_feature];
            }
            o_acc[query][cur_feature] =
                o_acc[query][cur_feature] * alpha[query] + sum;
        }

        __syncthreads();
    }

    // 写回.
    for (int i = tid; i < valid_query * d; i += stride) {
        //     int query_start = blockIdx.x * BR;
        int row = i / d;
        int col = i % d;
        int g_row = query_start + row;
        int g_col = col;
        output[g_row * d + g_col] = o_acc[row][col] / l[row];
    }
}
} // namespace

void attention_prefill::launch_query_tiled(const float *q, const float *k,
                                           const float *v, float *output,
                                           Problem problem,
                                           cudaStream_t stream) {
    if (problem.n <= 0 || problem.d <= 0 || problem.d > max_head_dimension) {
        throw std::invalid_argument("query-tiled requires N>0 and 1<=D<=128");
    }
    const std::int64_t element_count =
        static_cast<std::int64_t>(problem.n) * problem.d;
    if (element_count > std::numeric_limits<int>::max() ||
        problem.n > std::numeric_limits<int>::max() - (BC - 1)) {
        throw std::overflow_error(
            "query-tiled shape exceeds 32-bit device indexing");
    }
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr) {
        throw std::invalid_argument(
            "query-tiled requires non-null Q/K/V/output buffers");
    }

    const int grid = 1 + (problem.n - 1) / BR;
    query_tiled_kernel<<<grid, 128, 0, stream>>>(q, k, v, output, problem.n,
                                                 problem.d, problem.causal);
    GPU_CUDA_CHECK(cudaGetLastError());
}
