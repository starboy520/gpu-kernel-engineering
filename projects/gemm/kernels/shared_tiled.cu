#include "gemm/kernel.hpp"
#include "gpu_kernel/cuda_check.hpp"

constexpr int TILE_SIZE = 32;
__global__ void shared_tiled_kernel(const float *a, const float *b, float *c,
                                    int m, int n, int k) {

    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    // 每个线程计算一个C的位置
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < (k + TILE_SIZE - 1) / TILE_SIZE; i++) {

        // 当前线程加载A row 的所有列， 加载B col 列的所有行
        int a_col = i * TILE_SIZE + threadIdx.x;
        int b_row = i * TILE_SIZE + threadIdx.y;

        if (row < m && a_col < k) {
            tile_a[threadIdx.y][threadIdx.x] = a[row * k + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (b_row < k && col < n) {
            tile_b[threadIdx.y][threadIdx.x] = b[b_row * n + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 计算row, col
        for (int j = 0; j < TILE_SIZE; j++) {
            sum += tile_a[threadIdx.y][j] * tile_b[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

gemm::LaunchResult gemm::launch_shared_tiled(const float *a, const float *b,
                                             float *c, gemm::Problem problem,
                                             cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((block.x + problem.n - 1) / block.x,
              (block.y + problem.m - 1) / block.y);
    shared_tiled_kernel<<<grid, block, 0, stream>>>(a, b, c, problem.m,
                                                    problem.n, problem.k);
    GPU_CUDA_CHECK(cudaGetLastError());

    return {"shared", false};
}