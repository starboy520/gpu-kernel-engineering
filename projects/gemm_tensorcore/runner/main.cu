#include "gemm_tensorcore/kernel.hpp"
#include "gemm_tensorcore/reference.hpp"
#include "gpu_kernel/cuda_check.hpp"
#include "gpu_kernel/runner_utils.hpp"
#include "gpu_kernel/validation.hpp"

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

constexpr std::size_t kAElements =
    static_cast<std::size_t>(gemm_tensorcore::wmma_m) * gemm_tensorcore::wmma_k;
constexpr std::size_t kBElements =
    static_cast<std::size_t>(gemm_tensorcore::wmma_k) * gemm_tensorcore::wmma_n;
constexpr std::size_t kCElements =
    static_cast<std::size_t>(gemm_tensorcore::wmma_m) * gemm_tensorcore::wmma_n;

enum class InputPattern { ones, identity, random };

struct Options {
    InputPattern input = InputPattern::ones;
    std::uint32_t seed = 1234U;
};

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) {
        GPU_CUDA_CHECK(cudaMalloc(&pointer_, bytes));
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            (void)cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void *data() const { return pointer_; }

  private:
    void *pointer_ = nullptr;
};

const char *input_name(InputPattern input) {
    switch (input) {
    case InputPattern::ones:
        return "ones";
    case InputPattern::identity:
        return "identity";
    case InputPattern::random:
        return "random";
    }
    throw std::logic_error("unknown Tensor Core input pattern");
}

InputPattern parse_input(const std::string &value) {
    if (value == "ones") {
        return InputPattern::ones;
    }
    if (value == "identity") {
        return InputPattern::identity;
    }
    if (value == "random") {
        return InputPattern::random;
    }
    throw std::invalid_argument("--input must be ones, identity, or random");
}

Options parse_arguments(int argc, const char *const argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--input") {
            options.input = parse_input(
                gpu_kernel::require_value(argc, argv, index, option));
        } else if (option == "--seed") {
            options.seed = gpu_kernel::parse_integer<std::uint32_t>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    return options;
}

void prepare_inputs(InputPattern pattern, std::uint32_t seed,
                    std::vector<__half> &a, std::vector<__half> &b) {
    if (pattern == InputPattern::ones) {
        for (__half &value : a) {
            value = __float2half_rn(1.0F);
        }
        for (__half &value : b) {
            value = __float2half_rn(1.0F);
        }
        return;
    }

    if (pattern == InputPattern::identity) {
        for (int row = 0; row < gemm_tensorcore::wmma_m; ++row) {
            for (int column = 0; column < gemm_tensorcore::wmma_k; ++column) {
                a[static_cast<std::size_t>(row) * gemm_tensorcore::wmma_k +
                  column] = __float2half_rn(row == column ? 1.0F : 0.0F);
            }
        }
        for (int row = 0; row < gemm_tensorcore::wmma_k; ++row) {
            for (int column = 0; column < gemm_tensorcore::wmma_n; ++column) {
                const int code = (row * gemm_tensorcore::wmma_n + column) % 31;
                b[static_cast<std::size_t>(row) * gemm_tensorcore::wmma_n +
                  column] = __float2half_rn((code - 15) * 0.125F);
            }
        }
        return;
    }

    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    for (__half &value : a) {
        value = __float2half_rn(distribution(generator));
    }
    for (__half &value : b) {
        value = __float2half_rn(distribution(generator));
    }
}

int run(const Options &options) {
    std::vector<__half> a(kAElements);
    std::vector<__half> b(kBElements);
    std::vector<float> expected(kCElements);
    std::vector<float> actual(kCElements);
    prepare_inputs(options.input, options.seed, a, b);
    gemm_tensorcore::reference_wmma_single(a.data(), b.data(), expected.data());

    DeviceBuffer device_a(kAElements * sizeof(__half));
    DeviceBuffer device_b(kBElements * sizeof(__half));
    DeviceBuffer device_c(kCElements * sizeof(float));
    GPU_CUDA_CHECK(cudaMemcpy(device_a.data(), a.data(),
                              kAElements * sizeof(__half),
                              cudaMemcpyHostToDevice));
    GPU_CUDA_CHECK(cudaMemcpy(device_b.data(), b.data(),
                              kBElements * sizeof(__half),
                              cudaMemcpyHostToDevice));
    GPU_CUDA_CHECK(
        cudaMemset(device_c.data(), 0xFF, kCElements * sizeof(float)));

    gemm_tensorcore::launch_wmma_single(
        static_cast<const __half *>(device_a.data()),
        static_cast<const __half *>(device_b.data()),
        static_cast<float *>(device_c.data()), nullptr);
    GPU_CUDA_CHECK(cudaDeviceSynchronize());
    GPU_CUDA_CHECK(cudaMemcpy(actual.data(), device_c.data(),
                              kCElements * sizeof(float),
                              cudaMemcpyDeviceToHost));

    const gpu_kernel::ErrorMetrics metrics =
        gpu_kernel::compare(expected.data(), actual.data(), kCElements);
    const bool passed = gpu_kernel::passes(metrics, 1.0e-3, 1.0e-3);
    const std::size_t worst_row = metrics.worst_index / gemm_tensorcore::wmma_n;
    const std::size_t worst_column =
        metrics.worst_index % gemm_tensorcore::wmma_n;
    std::cout << std::fixed << std::setprecision(6)
              << "experiment=wmma-single shape=16x16x16"
              << " input=" << input_name(options.input)
              << " status=" << (passed ? "PASS" : "FAIL")
              << " max_abs=" << metrics.max_abs
              << " max_rel=" << metrics.max_rel << " worst=(" << worst_row
              << ',' << worst_column << ')' << " expected=" << metrics.expected
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
