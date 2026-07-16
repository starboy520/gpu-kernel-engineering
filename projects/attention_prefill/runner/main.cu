#include "attention_prefill/query_tiled.hpp"
#include "attention_prefill/warp_per_query.hpp"
#include "gpu_kernel/cuda_check.hpp"
#include "gpu_kernel/runner_utils.hpp"
#include "gpu_kernel/validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class InputPattern { random, zero_qk, rising_logits };
enum class Implementation { query_tiled, warp_per_query };

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) : pointer_(nullptr) {
        GPU_CUDA_CHECK(cudaMalloc(&pointer_, bytes));
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            (void)cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    float *data() const { return static_cast<float *>(pointer_); }

  private:
    void *pointer_;
};

struct Options {
    attention_prefill::Problem problem{0, 0, false};
    std::uint32_t seed = 1234U;
    InputPattern input_pattern = InputPattern::random;
    Implementation implementation = Implementation::query_tiled;
};

const char *implementation_name(Implementation implementation) {
    switch (implementation) {
    case Implementation::query_tiled:
        return "query-tiled";
    case Implementation::warp_per_query:
        return "warp-per-query";
    }
    throw std::logic_error("unknown implementation");
}

Implementation parse_implementation(const std::string &value) {
    if (value == "query-tiled") {
        return Implementation::query_tiled;
    }
    if (value == "warp-per-query") {
        return Implementation::warp_per_query;
    }
    throw std::invalid_argument(
        "--implementation must be query-tiled or warp-per-query");
}

const char *input_pattern_name(InputPattern pattern) {
    switch (pattern) {
    case InputPattern::random:
        return "random";
    case InputPattern::zero_qk:
        return "zero-qk";
    case InputPattern::rising_logits:
        return "rising-logits";
    }
    throw std::logic_error("unknown input pattern");
}

InputPattern parse_input_pattern(const std::string &value) {
    if (value == "random") {
        return InputPattern::random;
    }
    if (value == "zero-qk") {
        return InputPattern::zero_qk;
    }
    if (value == "rising-logits") {
        return InputPattern::rising_logits;
    }
    throw std::invalid_argument(
        "--input-pattern must be random, zero-qk, or rising-logits");
}

Options parse_arguments(int argc, const char *const argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--n") {
            options.problem.n = gpu_kernel::parse_integer<int>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--d") {
            options.problem.d = gpu_kernel::parse_integer<int>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--causal") {
            const int value = gpu_kernel::parse_integer<int>(
                gpu_kernel::require_value(argc, argv, index, option), option);
            if (value != 0 && value != 1) {
                throw std::invalid_argument("--causal must be 0 or 1");
            }
            options.problem.causal = value == 1;
        } else if (option == "--seed") {
            options.seed = gpu_kernel::parse_integer<std::uint32_t>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--input-pattern") {
            options.input_pattern = parse_input_pattern(
                gpu_kernel::require_value(argc, argv, index, option));
        } else if (option == "--implementation") {
            options.implementation = parse_implementation(
                gpu_kernel::require_value(argc, argv, index, option));
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (options.problem.n <= 0 || options.problem.d <= 0 ||
        options.problem.d > 128) {
        throw std::invalid_argument("require N>0 and 1<=D<=128");
    }
    return options;
}

std::vector<float> make_input(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    std::vector<float> values(count);
    for (float &value : values) {
        value = distribution(generator);
    }
    return values;
}

void apply_input_pattern(std::vector<float> &q, std::vector<float> &k,
                         attention_prefill::Problem problem,
                         InputPattern pattern) {
    if (pattern == InputPattern::random) {
        return;
    }
    if (pattern == InputPattern::zero_qk) {
        std::fill(q.begin(), q.end(), 0.0F);
        std::fill(k.begin(), k.end(), 0.0F);
        return;
    }

    std::fill(q.begin(), q.end(), 1.0F);
    for (int key = 0; key < problem.n; ++key) {
        const float value =
            static_cast<float>(key + 1) / static_cast<float>(problem.n);
        std::fill_n(k.begin() + static_cast<std::size_t>(key) * problem.d,
                    problem.d, value);
    }
}

void reference_attention(const std::vector<float> &q,
                         const std::vector<float> &k,
                         const std::vector<float> &v,
                         std::vector<float> &output,
                         attention_prefill::Problem problem) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(problem.d));
    std::vector<double> scores(static_cast<std::size_t>(problem.n));

    for (int query = 0; query < problem.n; ++query) {
        double row_max = -std::numeric_limits<double>::infinity();
        for (int key = 0; key < problem.n; ++key) {
            if (problem.causal && key > query) {
                scores[static_cast<std::size_t>(key)] =
                    -std::numeric_limits<double>::infinity();
                continue;
            }
            double dot = 0.0;
            for (int feature = 0; feature < problem.d; ++feature) {
                dot +=
                    static_cast<double>(
                        q[static_cast<std::size_t>(query) * problem.d +
                          feature]) *
                    static_cast<double>(
                        k[static_cast<std::size_t>(key) * problem.d + feature]);
            }
            const double score = dot * scale;
            scores[static_cast<std::size_t>(key)] = score;
            row_max = std::max(row_max, score);
        }

        double denominator = 0.0;
        for (int key = 0; key < problem.n; ++key) {
            double &score = scores[static_cast<std::size_t>(key)];
            if (std::isinf(score) && score < 0.0) {
                score = 0.0;
            } else {
                score = std::exp(score - row_max);
                denominator += score;
            }
        }

        for (int feature = 0; feature < problem.d; ++feature) {
            double accumulator = 0.0;
            for (int key = 0; key < problem.n; ++key) {
                accumulator +=
                    scores[static_cast<std::size_t>(key)] *
                    static_cast<double>(
                        v[static_cast<std::size_t>(key) * problem.d + feature]);
            }
            output[static_cast<std::size_t>(query) * problem.d + feature] =
                static_cast<float>(accumulator / denominator);
        }
    }
}

int run(const Options &options) {
    const std::size_t count = gpu_kernel::checked_multiply(
        static_cast<std::size_t>(options.problem.n),
        static_cast<std::size_t>(options.problem.d), "attention tensor");
    const std::size_t bytes =
        gpu_kernel::checked_multiply(count, sizeof(float), "attention bytes");

    std::vector<float> q = make_input(count, options.seed);
    std::vector<float> k = make_input(count, options.seed + 1U);
    const std::vector<float> v = make_input(count, options.seed + 2U);
    apply_input_pattern(q, k, options.problem, options.input_pattern);
    std::vector<float> expected(count);
    std::vector<float> actual(count);
    reference_attention(q, k, v, expected, options.problem);

    DeviceBuffer device_q(bytes);
    DeviceBuffer device_k(bytes);
    DeviceBuffer device_v(bytes);
    DeviceBuffer device_output(bytes);
    GPU_CUDA_CHECK(
        cudaMemcpy(device_q.data(), q.data(), bytes, cudaMemcpyHostToDevice));
    GPU_CUDA_CHECK(
        cudaMemcpy(device_k.data(), k.data(), bytes, cudaMemcpyHostToDevice));
    GPU_CUDA_CHECK(
        cudaMemcpy(device_v.data(), v.data(), bytes, cudaMemcpyHostToDevice));
    GPU_CUDA_CHECK(cudaMemset(device_output.data(), 0xFF, bytes));

    if (options.implementation == Implementation::query_tiled) {
        attention_prefill::launch_query_tiled(
            device_q.data(), device_k.data(), device_v.data(),
            device_output.data(), options.problem, nullptr);
    } else {
        attention_prefill::launch_warp_per_query(
            device_q.data(), device_k.data(), device_v.data(),
            device_output.data(), options.problem, nullptr);
    }
    GPU_CUDA_CHECK(cudaDeviceSynchronize());
    GPU_CUDA_CHECK(cudaMemcpy(actual.data(), device_output.data(), bytes,
                              cudaMemcpyDeviceToHost));

    const gpu_kernel::ErrorMetrics metrics =
        gpu_kernel::compare(expected.data(), actual.data(), count);
    const bool passed = gpu_kernel::passes(metrics, 2.0e-4, 2.0e-3);
    const std::size_t worst_query = metrics.worst_index / options.problem.d;
    const std::size_t worst_feature = metrics.worst_index % options.problem.d;
    std::cout << std::fixed << std::setprecision(6)
              << "kernel=" << implementation_name(options.implementation)
              << " shape=" << options.problem.n << 'x' << options.problem.d
              << " causal=" << (options.problem.causal ? 1 : 0)
              << " input_pattern=" << input_pattern_name(options.input_pattern)
              << " status=" << (passed ? "PASS" : "FAIL")
              << " max_abs=" << metrics.max_abs
              << " max_rel=" << metrics.max_rel << " worst=(" << worst_query
              << ',' << worst_feature << ") expected=" << metrics.expected
              << " actual=" << metrics.actual << '\n';
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, const char *const argv[]) {
    try {
        return run(parse_arguments(argc, argv));
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
