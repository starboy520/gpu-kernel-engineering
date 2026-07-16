#pragma once

#include "gemm_tensorcore/kernel.hpp"

namespace gemm_tensorcore {

// Reference consumes the already-quantized FP16 values and accumulates in
// FP32, matching the G0/G1 numerical contract.
void reference_wmma_single(const __half *a, const __half *b, float *c);

} // namespace gemm_tensorcore
