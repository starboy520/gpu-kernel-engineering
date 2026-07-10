#include "gemm/kernel.hpp"
#include "gemm/cuda_check.hpp"

__global__ void naive_kernel(const float* a,
                             const float* b,
                             float* c, int m, int n, int k) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < m && column < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + column];
        }
        c[row * n + column] = sum;
    }
}

gemm::LaunchResult gemm::launch_naive(
            const float* a,
            const float* b,
            float* c,
            gemm::Problem problem,
            cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((block.x+problem.n-1)/block.x, (block.y+problem.m-1)/block.y);
    naive_kernel<<<grid, block, 0, stream>>>(a, b, c, problem.m, problem.n, problem.k);
    CUDA_CHECK(cudaGetLastError());

    return {"naive", false};
}