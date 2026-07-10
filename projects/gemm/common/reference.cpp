#include "gemm/reference.hpp"

#include <cstddef>
#include <stdexcept>

namespace gemm {

void reference_cpu(const float* a, const float* b, float* c, int m, int n, int k) {
	if (a == nullptr || b == nullptr || c == nullptr) {
		throw std::invalid_argument("reference_cpu requires non-null matrices");
	}
	if (m <= 0 || n <= 0 || k <= 0) {
		throw std::invalid_argument("reference_cpu requires positive dimensions");
	}

	for (int row = 0; row < m; ++row) {
		for (int column = 0; column < n; ++column) {
			double accumulator = 0.0;
			for (int inner = 0; inner < k; ++inner) {
				const std::size_t a_index =
					static_cast<std::size_t>(row) * static_cast<std::size_t>(k) +
					static_cast<std::size_t>(inner);
				const std::size_t b_index =
					static_cast<std::size_t>(inner) * static_cast<std::size_t>(n) +
					static_cast<std::size_t>(column);
				accumulator += static_cast<double>(a[a_index]) *
							   static_cast<double>(b[b_index]);
			}
			const std::size_t c_index =
				static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
				static_cast<std::size_t>(column);
			c[c_index] = static_cast<float>(accumulator);
		}
	}
}

}  // namespace gemm
