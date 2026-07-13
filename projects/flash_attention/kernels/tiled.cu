#include "flash_attention/cuda_check.hpp"
#include "flash_attention/kernel.hpp"

namespace {

[[maybe_unused]] constexpr int BC = 16;
[[maybe_unused]] constexpr int MAX_D = 128;

[[maybe_unused]] __global__ void

// 一个block 负责一个 query， 一个block的线程负责一个query
/**
 *
 * 1. 协作加载k/v 一个 block的所有线程，s_k, s_v[BC][MAX_D]
 * 2. shared m, l 初始化 m = -INF, l = 0
 * 3. 计算score[i], 前16个线程 ， 每个线程都计算 一个query， scores  sshared
 * 4. 计算 m_new, l_new
 * 5. 对于key， 计算 xxxx
 * 6. scores[BC]:
 * 7. acc[MAX_D]： acc[a] = acc[a] * alpha
 *
 * 计算公式：
 * score_j = (sum_x Q[query, x] * K[j, x]) / sqrt(d)
 * causal && j > query 时，score_j = -INFINITY
 *
 * m_new = max(m, max_j(score_j))
 * alpha = (m == -INFINITY) ? 0 : exp(m - m_new)
 * weight_j = exp(score_j - m_new)
 * l_new = alpha * l + sum_j(weight_j)
 * acc_new[x] = alpha * acc[x] + sum_j(weight_j * V[j, x])
 *
 * 全部 key 处理完成后：
 * output[query, x] = acc[x] / l
 */
tiled_attention_kernel(const float *q, const float *k, const float *v,
                       float *output, int n, int d, bool causal) {
    __shared__ float s_q[MAX_D];

    __shared__ float s_k[BC][MAX_D];
    __shared__ float s_v[BC][MAX_D];
    __shared__ float m;
    __shared__ float l;
    __shared__ float score[BC];
    __shared__ float alpha;
    __shared__ float acc[MAX_D];

    float scale = rsqrt((float)d);

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
    while (start < n) {
        int valid = min(BC, n - start);
        for (int i = tid; i < valid * d; i += stride) {
            int key = i / d;
            int feature = i % d;
            s_k[key][feature] = k[(start + key) * d + feature];
            s_v[key][feature] = v[(start + key) * d + feature];
        }
        __syncthreads();
        // 搬运完成
        // 下一步计算score
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
                    sum += s_q[i] * s_k[cur_key][i];
                }

                score[cur_key] = sum * scale;
            }
        }
        __syncthreads();

        // 对当前 key tile 做 online softmax 更新。
        // score[] 在一个 tile 内分两个阶段复用：
        //   1. 上面先保存原始 logit：Q[query] 与 K[start + i] 的点积；
        //   2. 下面再原地改写为未归一化权重 exp(logit - m_new)。
        // 原始 logit 在算出当前 tile 的最大值后便不再需要，因此复用同一块
        // shared memory；本 tile 更新完 acc 后，下一个 tile 会重新覆盖 score[]。
        //
        // acc[feature] 不是“某个 key 的累加值”，而是当前 query 在已经处理过的
        // 所有 key 上，对 V[:, feature] 的未归一化加权和（softmax 分子）：
        //   acc_new[x] = alpha * acc_old[x]
        //              + sum_i(score[i] * V[start + i, x])
        // 其中 alpha = exp(m_old - m_new)，用于把旧分子缩放到新的全局最大值
        // m_new 所对应的指数基准；l 以相同方式更新，最后 output[x] = acc[x] / l。
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
        __syncthreads();

        for (int feature = threadIdx.x; feature < d; feature += blockDim.x) {
            float add = 0.0;
            for (int i = 0; i < valid; i++) {
                add += score[i] * s_v[i][feature];
            }
            acc[feature] = acc[feature] * alpha + add;
        }
        __syncthreads();
        start += valid;
    }

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        o_row[i] = acc[i] / l;
    }
}

} // namespace

flash_attention::LaunchResult flash_attention::launch_tiled_online(
    const float *q, const float *k, const float *v, float *output,
    float *workspace, Problem problem, cudaStream_t stream) {
    (void)q;
    (void)k;
    (void)v;
    (void)output;
    (void)workspace;
    (void)problem;
    (void)stream;
    int grid = problem.n;

    tiled_attention_kernel<<<grid, 128, 0, stream>>>(q, k, v, output, problem.n,
                                                     problem.d, problem.causal);
    FA_CUDA_CHECK(cudaGetLastError());
    return {"tiled", false};
}
