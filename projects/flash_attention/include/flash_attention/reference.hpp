#pragma once

#include "flash_attention/kernel.hpp"

namespace flash_attention {

// Direct mathematical reference. Dot products, exponentials and output
// accumulation use double precision before the final float conversion.
void reference_cpu(const float *q, const float *k, const float *v,
                   float *output, Problem problem);

} // namespace flash_attention