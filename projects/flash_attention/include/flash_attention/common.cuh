constexpr int MAX_THREAD_PER_BLOCK = 1024;
constexpr int MAX_THREAD_PER_WARP = 32;

__device__ __forceinline__ float warpReduceSumF(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

__device__ __forceinline__ float blockReduceSumF(float value) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    __shared__ float s[MAX_THREAD_PER_BLOCK / MAX_THREAD_PER_WARP];
    float sum = warpReduceSumF(value);
    if (lane_id == 0) {
        s[warp_id] = sum;
    }
    __syncthreads();

    // 这里需要保证block线程数是32的倍数，
    int num_warp = blockDim.x / MAX_THREAD_PER_WARP;

    value = lane_id < num_warp ? s[lane_id] : 0.0f;
    if (warp_id == 0) {
        value = warpReduceSumF(value);
    }
    return value;
}

__device__ __forceinline__ float warpReduceMaxF(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

__device__ __forceinline__ float blockReduceMaxF(float value) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    __shared__ float s[MAX_THREAD_PER_BLOCK / MAX_THREAD_PER_WARP];
    float cur_max = warpReduceMaxF(value);
    if (lane_id == 0) {
        s[warp_id] = cur_max;
    }
    __syncthreads();

    // 这里需要保证block线程数是32的倍数，
    int num_warp = blockDim.x / MAX_THREAD_PER_WARP;

    value = lane_id < num_warp ? s[lane_id] : -INFINITY;
    if (warp_id == 0) {
        value = warpReduceMaxF(value);
    }
    return value;
}
