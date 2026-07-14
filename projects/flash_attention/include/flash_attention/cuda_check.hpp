#pragma once

#include "gpu_kernel/cuda_check.hpp"

namespace flash_attention {

using gpu_kernel::throw_cuda_error;

} // namespace flash_attention

#define FA_CUDA_CHECK(expr) GPU_CUDA_CHECK(expr)
