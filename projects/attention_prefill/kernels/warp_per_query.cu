#include "attention_prefill/warp_per_query.hpp"
#include "gpu_kernel/cuda_check.hpp"

#include <stdexcept>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_CTA = 128;
constexpr int BC = 16;
constexpr int MAX_D = 128;
// constexpr int FRAGMENT_SLOTS = MAX_D / 32;
constexpr int FEATURE_PER_LANE = MAX_D / 32;
constexpr int QUERY_PER_BLOCK = THREADS_PER_CTA / WARP_SIZE;

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

__global__ void warp_per_query_kernel(const float *q, const float *k,
                                      const float *v, float *output, int n,
                                      int d, bool causal) {
    // CTA 共享一份 K/V tile；四个 Warp 分别消费它。
    __shared__ float k_s[BC][MAX_D];
    __shared__ float v_s[BC][MAX_D];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // 每个 lane 只  load 对应的
    float o_acc[FEATURE_PER_LANE];
    float feature[FEATURE_PER_LANE];
    for (int i = 0; i < FEATURE_PER_LANE; i++) {
        o_acc[i] = 0.0f;
        feature[i] = 0.0f;
    }

    int stride = blockDim.x;
    int tid = threadIdx.x;

    // 1. load query feature, 每个warp的 lane 只处理对应的 feature
    int query_base = blockIdx.x * QUERY_PER_BLOCK;
    if (query_base + warp_id >= n) {
        for (int i = 0; i < FEATURE_PER_LANE; i++) {
            feature[i] = 0.0f;
        }
    } else {
        const float *query = q + (query_base + warp_id) * d;
        for (int i = lane_id; i < d; i += WARP_SIZE) {
            feature[i / WARP_SIZE] = query[i];
        }
    }

    float m = -INFINITY;
    float l = 0.0f;
    for (int kv_start = 0; kv_start < n; kv_start += BC) {
        int valid_kv = min(BC, n - kv_start);

        // cta 协作 load
        load_data(&k_s[0][0], MAX_D, valid_kv, d, k, kv_start, 0, d, tid,
                  stride);
        load_data(&v_s[0][0], MAX_D, valid_kv, d, v, kv_start, 0, d, tid,
                  stride);
        __syncthreads();

        // 计算 score, warp内协作，
        //  一个 warp 负责一个query，
        // 一个lane_id 存一个 key 对应的owned_score 其实也就是weight ,
        //  TODO: 如果 bc > 32 怎么办? 32怎么办
        // 但是计算的时候是 warp 所有的lane_id协作协作
        float owned_score = -INFINITY;
        float alpha = 1.0f;
        for (int key = 0; key < valid_kv; key++) {
            int g_query = query_base + warp_id;
            int g_kv = kv_start + key;
            if (g_kv > g_query && causal) {
                continue;
            }
            float sum = 0.0f;
            for (int i = 0; i < FEATURE_PER_LANE; i++) {
                int f_dim = lane_id + 32 * i;
                if (f_dim < d) {
                    sum += feature[i] * k_s[key][f_dim];
                }
            }

            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            }
            float warp_sum = __shfl_sync(0xffffffff, sum, 0);
            if (lane_id == key) {
                owned_score = warp_sum * rsqrtf((float)d);
            }
        }

        float tile_m = owned_score;
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_m =
                fmaxf(tile_m, __shfl_down_sync(0xffffffffu, tile_m, offset));
        }

        // 广播给所有lane
        tile_m = __shfl_sync(0xffffffffu, tile_m, 0);

        float tile_l = 0.0f;
        float weight = 0.0f;

        float m_new = fmaxf(tile_m, m);
        if (isinf(tile_m) && tile_m < 0.0f) {
            alpha = 1.0f; // 本 tile 没有贡献, weight 也还是0
        } else {
            alpha = expf(m - m_new);
            weight = expf(owned_score - m_new);
        }
        tile_l = weight;
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_l += __shfl_down_sync(0xffffffffu, tile_l, offset);
        }
        if (lane_id == 0) {
            l = l * alpha + tile_l;
        }
        l = __shfl_sync(0xffffffffu, l, 0);
        m = m_new;

        // 接下来计算occ
        // occ[key][feature] += sum(weight * vs[i][feature])
        // weight score[query][all_key] 就是所有的 一个lane所有的weight
        for (int f_idx = 0; f_idx < FEATURE_PER_LANE; f_idx++) {
            float occ_sum = 0.0f;
            for (int i = 0; i < valid_kv; i++) {
                float cur = __shfl_sync(0xffffffffu, weight, i);
                int g_feature_dim = lane_id + f_idx * 32;
                if (g_feature_dim < d) {
                    occ_sum += cur * v_s[i][g_feature_dim];
                }
            }
            o_acc[f_idx] = o_acc[f_idx] * alpha + occ_sum;
        }
        __syncthreads();
    }

    // 写回
    for (int i = 0; i < FEATURE_PER_LANE; i++) {
        int col = i * 32 + lane_id;
        int g_row = query_base + warp_id;
        if (col < d && g_row < n) {
            output[g_row * d + col] = o_acc[i] / l;
        }
    }
}

} // namespace

void attention_prefill::launch_warp_per_query(const float *q, const float *k,
                                              const float *v, float *output,
                                              Problem problem,
                                              cudaStream_t stream) {
    if (problem.n <= 0 || problem.d <= 0 || problem.d > max_head_dimension) {
        throw std::invalid_argument(
            "warp-per-query requires N>0 and 1<=D<=128");
    }
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr) {
        throw std::invalid_argument(
            "warp-per-query requires non-null Q/K/V/output buffers");
    }

    const int grid = (problem.n - 1 + m2_queries_per_cta) / m2_queries_per_cta;
    warp_per_query_kernel<<<grid, THREADS_PER_CTA, 0, stream>>>(
        q, k, v, output, problem.n, problem.d, problem.causal);
    GPU_CUDA_CHECK(cudaGetLastError());
}
