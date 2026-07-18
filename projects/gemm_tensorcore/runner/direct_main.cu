#include "gemm_tensorcore/kernel.hpp"
#include "gemm_tensorcore/reference.hpp"
#include "gpu_kernel/cuda_check.hpp"
#include "gpu_kernel/runner_utils.hpp"
#include "gpu_kernel/validation.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class InputPattern { ones, identity, random };

struct Options {
    gemm_tensorcore::Problem problem{16, 16, 16};
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
    throw std::logic_error("unknown Direct WMMA input pattern");
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
        if (option == "--m") {
            options.problem.m = gpu_kernel::parse_integer<int>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--n") {
            options.problem.n = gpu_kernel::parse_integer<int>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--k") {
            options.problem.k = gpu_kernel::parse_integer<int>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--input") {
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

std::size_t matrix_elements(int rows, int columns, const char *description) {
    return gpu_kernel::checked_multiply(static_cast<std::size_t>(rows),
                                        static_cast<std::size_t>(columns),
                                        description);
}

void set_b_col_major(std::vector<__half> &b, int ldb, int inner, int column,
                     float value) {
    b[static_cast<std::size_t>(inner) +
      static_cast<std::size_t>(column) * ldb] = __float2half_rn(value);
}

void prepare_inputs(InputPattern pattern, std::uint32_t seed,
                    gemm_tensorcore::Problem logical,
                    gemm_tensorcore::Problem padded, std::vector<__half> &a,
                    std::vector<__half> &b) {
    if (pattern == InputPattern::ones) {
        for (int row = 0; row < logical.m; ++row) {
            for (int inner = 0; inner < logical.k; ++inner) {
                a[static_cast<std::size_t>(row) * padded.k + inner] =
                    __float2half_rn(1.0F);
            }
        }
        for (int column = 0; column < logical.n; ++column) {
            for (int inner = 0; inner < logical.k; ++inner) {
                set_b_col_major(b, padded.k, inner, column, 1.0F);
            }
        }
        return;
    }

    if (pattern == InputPattern::identity) {
        for (int row = 0; row < logical.m; ++row) {
            for (int inner = 0; inner < logical.k; ++inner) {
                a[static_cast<std::size_t>(row) * padded.k + inner] =
                    __float2half_rn(row == inner ? 1.0F : 0.0F);
            }
        }
        for (int column = 0; column < logical.n; ++column) {
            for (int inner = 0; inner < logical.k; ++inner) {
                const int code = (inner * logical.n + column) % 31;
                set_b_col_major(b, padded.k, inner, column,
                                (code - 15) * 0.125F);
            }
        }
        return;
    }

    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    for (int row = 0; row < logical.m; ++row) {
        for (int inner = 0; inner < logical.k; ++inner) {
            a[static_cast<std::size_t>(row) * padded.k + inner] =
                __float2half_rn(distribution(generator));
        }
    }
    for (int column = 0; column < logical.n; ++column) {
        for (int inner = 0; inner < logical.k; ++inner) {
            set_b_col_major(b, padded.k, inner, column,
                            distribution(generator));
        }
    }
}

int run(const Options &options) {
    const gemm_tensorcore::Problem padded =
        gemm_tensorcore::pad_problem(options.problem);
    const std::size_t a_count =
        matrix_elements(padded.m, padded.k, "Direct WMMA padded A");
    const std::size_t b_count =
        matrix_elements(padded.k, padded.n, "Direct WMMA padded B");
    const std::size_t c_count =
        matrix_elements(padded.m, padded.n, "Direct WMMA padded C");
    const std::size_t a_bytes = gpu_kernel::checked_multiply(
        a_count, sizeof(__half), "Direct WMMA padded A bytes");
    const std::size_t b_bytes = gpu_kernel::checked_multiply(
        b_count, sizeof(__half), "Direct WMMA padded B bytes");
    const std::size_t c_bytes = gpu_kernel::checked_multiply(
        c_count, sizeof(float), "Direct WMMA padded C bytes");

    std::vector<__half> a(a_count, __float2half_rn(0.0F));
    std::vector<__half> b(b_count, __float2half_rn(0.0F));
    std::vector<float> expected(c_count);
    std::vector<float> actual(c_count);
    prepare_inputs(options.input, options.seed, options.problem, padded, a, b);
    gemm_tensorcore::reference_wmma_direct(a.data(), padded.k, b.data(),
                                           padded.k, expected.data(), padded.n,
                                           padded);

    DeviceBuffer device_a(a_bytes);
    DeviceBuffer device_b(b_bytes);
    DeviceBuffer device_c(c_bytes);
    GPU_CUDA_CHECK(
        cudaMemcpy(device_a.data(), a.data(), a_bytes, cudaMemcpyHostToDevice));
    GPU_CUDA_CHECK(
        cudaMemcpy(device_b.data(), b.data(), b_bytes, cudaMemcpyHostToDevice));
    GPU_CUDA_CHECK(cudaMemset(device_c.data(), 0xFF, c_bytes));

    gemm_tensorcore::launch_wmma_direct(
        static_cast<const __half *>(device_a.data()),
        static_cast<const __half *>(device_b.data()),
        static_cast<float *>(device_c.data()), padded, nullptr);
    GPU_CUDA_CHECK(cudaDeviceSynchronize());
    GPU_CUDA_CHECK(cudaMemcpy(actual.data(), device_c.data(), c_bytes,
                              cudaMemcpyDeviceToHost));

    const gpu_kernel::ErrorMetrics metrics =
        gpu_kernel::compare(expected.data(), actual.data(), c_count);
    const bool passed = gpu_kernel::passes(metrics, 1.0e-3, 1.0e-3);
    const std::size_t worst_row =
        metrics.worst_index / static_cast<std::size_t>(padded.n);
    const std::size_t worst_column =
        metrics.worst_index % static_cast<std::size_t>(padded.n);

    std::cout << std::fixed << std::setprecision(6) << "experiment=wmma-direct"
              << " shape=" << options.problem.m << 'x' << options.problem.n
              << 'x' << options.problem.k << " padded=" << padded.m << 'x'
              << padded.n << 'x' << padded.k << " layout=A.row,B.col,C.row"
              << " input=" << input_name(options.input)
              << " status=" << (passed ? "PASS" : "FAIL")
              << " max_abs=" << metrics.max_abs
              << " max_rel=" << metrics.max_rel << " worst=(" << worst_row
              << ',' << worst_column << ") expected=" << metrics.expected
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
