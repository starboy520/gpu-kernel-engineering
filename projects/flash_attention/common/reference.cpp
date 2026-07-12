#include "flash_attention/reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace flash_attention {

void reference_cpu(const float* q, const float* k, const float* v,
                   float* output, Problem problem) {
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr) {
        throw std::invalid_argument("reference_cpu requires non-null tensors");
    }
    if (problem.n <= 0 || problem.d <= 0) {
        throw std::invalid_argument("reference_cpu requires positive N and D");
    }

    const int n = problem.n;
    const int d = problem.d;
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    std::vector<double> scores(static_cast<std::size_t>(n));

    for (int query = 0; query < n; ++query) {
        double row_max = -std::numeric_limits<double>::infinity();
        for (int key = 0; key < n; ++key) {
            if (problem.causal && key > query) {
                scores[static_cast<std::size_t>(key)] =
                    -std::numeric_limits<double>::infinity();
                continue;
            }

            double dot = 0.0;
            for (int feature = 0; feature < d; ++feature) {
                const std::size_t q_index =
                    static_cast<std::size_t>(query) * d + feature;
                const std::size_t k_index =
                    static_cast<std::size_t>(key) * d + feature;
                dot += static_cast<double>(q[q_index]) *
                       static_cast<double>(k[k_index]);
            }
            const double score = dot * scale;
            scores[static_cast<std::size_t>(key)] = score;
            row_max = std::max(row_max, score);
        }

        double denominator = 0.0;
        for (int key = 0; key < n; ++key) {
            double& score = scores[static_cast<std::size_t>(key)];
            score = std::exp(score - row_max);
            denominator += score;
        }

        for (int feature = 0; feature < d; ++feature) {
            double accumulator = 0.0;
            for (int key = 0; key < n; ++key) {
                const std::size_t v_index =
                    static_cast<std::size_t>(key) * d + feature;
                accumulator +=
                    (scores[static_cast<std::size_t>(key)] / denominator) *
                    static_cast<double>(v[v_index]);
            }
            const std::size_t output_index =
                static_cast<std::size_t>(query) * d + feature;
            output[output_index] = static_cast<float>(accumulator);
        }
    }
}

}  // namespace flash_attention