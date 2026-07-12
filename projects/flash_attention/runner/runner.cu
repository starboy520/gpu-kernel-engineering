#include "flash_attention/runner.hpp"

#include "flash_attention/cuda_check.hpp"
#include "flash_attention/reference.hpp"
#include "flash_attention/validation.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace flash_attention {
namespace {

constexpr double kValidationAtol = 2.0e-4;
constexpr double kValidationRtol = 2.0e-3;

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) : pointer_(nullptr) {
        FA_CUDA_CHECK(cudaMalloc(&pointer_, bytes));
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            (void)cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    float *data() { return static_cast<float *>(pointer_); }

  private:
    void *pointer_;
};

class EventHandle {
  public:
    EventHandle() : event_(nullptr) { FA_CUDA_CHECK(cudaEventCreate(&event_)); }

    ~EventHandle() {
        if (event_ != nullptr) {
            (void)cudaEventDestroy(event_);
        }
    }

    cudaEvent_t get() const { return event_; }

  private:
    cudaEvent_t event_;
};

std::size_t checked_multiply_impl(std::size_t left, std::size_t right,
                                  std::string_view description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error("size overflow while computing " +
                                  std::string(description));
    }
    return left * right;
}

std::size_t tensor_count(Problem problem) {
    return checked_multiply_impl(static_cast<std::size_t>(problem.n),
                                 static_cast<std::size_t>(problem.d),
                                 "attention tensor");
}

std::size_t float_bytes(std::size_t count, const char *description) {
    return checked_multiply_impl(count, sizeof(float), description);
}

std::vector<float> generate_input_impl(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    std::vector<float> values(count);
    for (float &value : values) {
        value = distribution(generator);
    }
    return values;
}

std::string require_value(int argc, const char *const argv[], int &index,
                          const std::string &option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + option);
    }
    return argv[++index];
}

template <typename Integer>
Integer parse_integer(const std::string &text, const std::string &option) {
    Integer value{};
    const char *begin = text.data();
    const char *end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec == std::errc::result_out_of_range) {
        throw std::invalid_argument("integer overflow for " + option + ": " +
                                    text);
    }
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument("invalid integer for " + option + ": " +
                                    text);
    }
    return value;
}

const char *path_or_name(const LaunchResult &result,
                         const KernelDescriptor &kernel) {
    return result.selected_path == nullptr || result.selected_path[0] == '\0'
               ? kernel.name
               : result.selected_path;
}

void print_result(std::ostream &output, const KernelDescriptor &kernel,
                  const char *path, Problem problem, bool passed,
                  const ErrorMetrics &metrics, double latency_ms,
                  std::size_t workspace_bytes) {
    const std::size_t worst_query =
        metrics.worst_index / static_cast<std::size_t>(problem.d);
    const std::size_t worst_feature =
        metrics.worst_index % static_cast<std::size_t>(problem.d);
    output << std::fixed << std::setprecision(6) << "kernel=" << kernel.name
           << " path=" << path << " shape=" << problem.n << 'x' << problem.d
           << " causal=" << (problem.causal ? 1 : 0)
           << " status=" << (passed ? "PASS" : "FAIL")
           << " max_abs=" << metrics.max_abs << " max_rel=" << metrics.max_rel
           << " worst=(" << worst_query << ',' << worst_feature
           << ") expected=" << metrics.expected << " actual=" << metrics.actual
           << " latency_ms=" << latency_ms
           << " workspace_bytes=" << workspace_bytes << '\n';
}

} // namespace

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             std::string_view description) {
    return checked_multiply_impl(left, right, description);
}

std::vector<float> generate_input(std::size_t count, std::uint32_t seed) {
    return generate_input_impl(count, seed);
}

RunnerOptions parse_arguments(int argc, const char *const argv[]) {
    RunnerOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help") {
            options.help = true;
        } else if (option == "--list") {
            options.list = true;
        } else if (option == "--kernel") {
            options.kernel = require_value(argc, argv, index, option);
        } else if (option == "--n") {
            options.problem.n = parse_integer<int>(
                require_value(argc, argv, index, option), option);
        } else if (option == "--d") {
            options.problem.d = parse_integer<int>(
                require_value(argc, argv, index, option), option);
        } else if (option == "--causal") {
            const int causal = parse_integer<int>(
                require_value(argc, argv, index, option), option);
            if (causal != 0 && causal != 1) {
                throw std::invalid_argument("--causal must be 0 or 1");
            }
            options.problem.causal = causal == 1;
        } else if (option == "--mode") {
            const std::string mode = require_value(argc, argv, index, option);
            if (mode == "validate") {
                options.mode = RunMode::validate;
            } else if (mode == "benchmark") {
                options.mode = RunMode::benchmark;
            } else {
                throw std::invalid_argument(
                    "--mode must be validate or benchmark");
            }
        } else if (option == "--warmup") {
            options.warmup = parse_integer<int>(
                require_value(argc, argv, index, option), option);
        } else if (option == "--iterations") {
            options.iterations = parse_integer<int>(
                require_value(argc, argv, index, option), option);
        } else if (option == "--seed") {
            options.seed = parse_integer<std::uint32_t>(
                require_value(argc, argv, index, option), option);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    return options;
}

void validate_options(const RunnerOptions &options) {
    if (options.help || options.list) {
        return;
    }
    if (options.kernel.empty()) {
        throw std::invalid_argument("normal run requires --kernel");
    }
    if (options.problem.n <= 0 || options.problem.d <= 0) {
        throw std::invalid_argument("--n and --d must be positive integers");
    }
    if (options.problem.d > 128) {
        throw std::invalid_argument("--d must be <= 128 in the first version");
    }
    if (options.warmup < 0 || options.iterations <= 0) {
        throw std::invalid_argument(
            "--warmup must be nonnegative and --iterations positive");
    }
}

int run(const RunnerOptions &options, std::ostream &output) {
    validate_options(options);
    if (options.help) {
        output << "Usage: flash_attention_runner [options]\n"
               << "  --help                         Show this message\n"
               << "  --list                         List registered kernels\n"
               << "  --kernel <name>                Kernel to run\n"
               << "  --n <int>                      Sequence length\n"
               << "  --d <int>                      Single-head dimension\n"
               << "  --causal <0|1>                 Default: 0\n"
               << "  --mode <validate|benchmark>    Default: validate\n"
               << "  --warmup <count>               Default: 5\n"
               << "  --iterations <count>           Default: 20\n"
               << "  --seed <uint>                  Default: 1234\n";
        return 0;
    }
    if (options.list) {
        for (const KernelDescriptor &kernel : registered_kernels()) {
            output << kernel.name << '\n';
        }
        return 0;
    }

    const KernelDescriptor *kernel = find_kernel(options.kernel);
    if (kernel == nullptr) {
        throw std::invalid_argument("unknown kernel: " + options.kernel);
    }

    const std::size_t count = tensor_count(options.problem);
    const std::size_t bytes = float_bytes(count, "Q/K/V/output");
    const std::size_t workspace_bytes =
        kernel->workspace_bytes(options.problem);

    const std::vector<float> host_q = generate_input(count, options.seed);
    const std::vector<float> host_k = generate_input(count, options.seed + 1U);
    const std::vector<float> host_v = generate_input(count, options.seed + 2U);
    std::vector<float> expected(count);
    std::vector<float> actual(count);
    reference_cpu(host_q.data(), host_k.data(), host_v.data(), expected.data(),
                  options.problem);

    DeviceBuffer device_q(bytes);
    DeviceBuffer device_k(bytes);
    DeviceBuffer device_v(bytes);
    DeviceBuffer device_output(bytes);
    DeviceBuffer device_workspace(workspace_bytes);

    FA_CUDA_CHECK(cudaMemcpy(device_q.data(), host_q.data(), bytes,
                             cudaMemcpyHostToDevice));
    FA_CUDA_CHECK(cudaMemcpy(device_k.data(), host_k.data(), bytes,
                             cudaMemcpyHostToDevice));
    FA_CUDA_CHECK(cudaMemcpy(device_v.data(), host_v.data(), bytes,
                             cudaMemcpyHostToDevice));
    // 0xFF is a NaN bit pattern for float. An unimplemented or incomplete
    // kernel therefore fails validation instead of accidentally passing.
    FA_CUDA_CHECK(cudaMemset(device_output.data(), 0xFF, bytes));
    FA_CUDA_CHECK(cudaMemset(device_workspace.data(), 0xFF, workspace_bytes));

    const LaunchResult validation_launch = kernel->launch(
        device_q.data(), device_k.data(), device_v.data(), device_output.data(),
        device_workspace.data(), options.problem, nullptr);
    FA_CUDA_CHECK(cudaPeekAtLastError());
    FA_CUDA_CHECK(cudaDeviceSynchronize());
    FA_CUDA_CHECK(cudaMemcpy(actual.data(), device_output.data(), bytes,
                             cudaMemcpyDeviceToHost));

    const ErrorMetrics metrics = compare(expected.data(), actual.data(), count);
    const bool passed = passes(metrics, kValidationAtol, kValidationRtol);
    const char *path = path_or_name(validation_launch, *kernel);

    if (options.mode == RunMode::validate || !passed) {
        print_result(output, *kernel, path, options.problem, passed, metrics,
                     0.0, workspace_bytes);
        return passed ? 0 : 1;
    }

    for (int warmup = 0; warmup < options.warmup; ++warmup) {
        (void)kernel->launch(device_q.data(), device_k.data(), device_v.data(),
                             device_output.data(), device_workspace.data(),
                             options.problem, nullptr);
    }
    FA_CUDA_CHECK(cudaDeviceSynchronize());

    EventHandle start;
    EventHandle stop;
    FA_CUDA_CHECK(cudaEventRecord(start.get()));
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        (void)kernel->launch(device_q.data(), device_k.data(), device_v.data(),
                             device_output.data(), device_workspace.data(),
                             options.problem, nullptr);
    }
    FA_CUDA_CHECK(cudaEventRecord(stop.get()));
    FA_CUDA_CHECK(cudaEventSynchronize(stop.get()));
    float total_ms = 0.0F;
    FA_CUDA_CHECK(cudaEventElapsedTime(&total_ms, start.get(), stop.get()));
    const double latency_ms =
        static_cast<double>(total_ms) / options.iterations;
    print_result(output, *kernel, path, options.problem, true, metrics,
                 latency_ms, workspace_bytes);
    return 0;
}

} // namespace flash_attention