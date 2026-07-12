#include "flash_attention/common.cuh"
#include "flash_attention/cuda_check.hpp"
#include "flash_attention/kernel.hpp"

// Student-owned file: implement the three kernels and their launch geometry.
// The runner supplies Q/K/V/output plus an N*N scores workspace.

__global__ void qk_scores(const float *q, const float *k, float *scores, int n,
                          int d, bool causal) {
    // one thread computes one score[query, key].
    int query = blockDim.y * blockIdx.y + threadIdx.y;
    int key = blockDim.x * blockIdx.x + threadIdx.x;
    float c = rsqrtf((float)d);
    if (query < n && key < n) {
        if (key > query && causal) {
            scores[query * n + key] = -INFINITY;
        } else {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                // 这里注意 k 也是n *d
                sum += q[query * d + i] * k[key * d + i];
            }
            scores[query * n + key] = sum * c;
        }
    }
}

__global__ void row_softmax(float *scores, int n) {
    // 一个block 负责一行
    __shared__ float row_max;
    __shared__ float row_sum;

    int tid = threadIdx.x;
    float *row_s = scores + blockIdx.x * n;
    float t_max = -INFINITY;

    int stride = blockDim.x;
    for (int i = tid; i < n; i += stride) {
        t_max = fmaxf(t_max, row_s[i]);
    }
    t_max = blockReduceMaxF(t_max);

    if (threadIdx.x == 0) {
        row_max = t_max;
    }
    __syncthreads();
    float t_sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        t_sum += expf(row_s[i] - row_max);
    }
    t_sum = blockReduceSumF(t_sum);
    if (threadIdx.x == 0) {
        row_sum = t_sum;
    }
    __syncthreads();
    //
    for (int i = tid; i < n; i += stride) {
        row_s[i] = expf(row_s[i] - row_max) / row_sum;
    }
}

__global__ void pv_output(const float *probabilities, const float *v,
                          float *output, int n, int d) {
    // TODO(student): one thread computes one output[query, feature].

    int query = blockDim.y * blockIdx.y + threadIdx.y;
    int feature = blockDim.x * blockIdx.x + threadIdx.x;
    if (query < n && feature < d) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += probabilities[query * n + i] * v[i * d + feature];
        }
        output[query * d + feature] = sum;
    }
}

flash_attention::LaunchResult flash_attention::launch_naive_materialized(
    const float *q, const float *k, const float *v, float *output,
    float *scores, Problem problem, cudaStream_t stream) {
    // TODO(student): choose grid/block shapes and launch, in order:
    //   1. qk_scores
    //   2. row_softmax
    //   3. pv_output
    // Then check the asynchronous launch status and return {"naive", false}.

    dim3 block(32, 32);
    dim3 grid((block.x + problem.n - 1) / block.x,
              (block.y + problem.n - 1) / block.y);
    qk_scores<<<grid, block, 0, stream>>>(q, k, scores, problem.n, problem.d,
                                          problem.causal);
    FA_CUDA_CHECK(cudaGetLastError());
    row_softmax<<<problem.n, 32, 0, stream>>>(scores, problem.n);
    FA_CUDA_CHECK(cudaGetLastError());
    dim3 grid2((block.x + problem.d - 1) / block.x,
               (block.y + problem.n - 1) / block.y);
    pv_output<<<grid2, block, 0, stream>>>(scores, v, output, problem.n,
                                           problem.d);
    FA_CUDA_CHECK(cudaGetLastError());
    return {"naive", false};
}