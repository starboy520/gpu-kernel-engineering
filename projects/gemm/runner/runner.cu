#include "gemm/runner.hpp"

#include "gemm/cuda_check.hpp"
#include "gemm/kernel.hpp"
#include "gemm/reference.hpp"
#include "gemm/validation.hpp"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>

namespace gemm {
namespace {

constexpr double kValidationAtol = 1.0e-3;
constexpr double kValidationRtol = 1.0e-3;

cudaError_t default_malloc(void** pointer, std::size_t bytes) {
    return cudaMalloc(pointer, bytes);
}

cudaError_t default_free(void* pointer) {
    return cudaFree(pointer);
}

cudaError_t default_memcpy_h2d(void* destination, const void* source, std::size_t bytes) {
    return cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);
}

cudaError_t default_memcpy_d2h(void* destination, const void* source, std::size_t bytes) {
    return cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);
}

cudaError_t default_peek_at_last_error() {
    return cudaPeekAtLastError();
}

cudaError_t default_device_synchronize() {
    return cudaDeviceSynchronize();
}

cudaError_t default_event_create(cudaEvent_t* event) {
    return cudaEventCreate(event);
}

cudaError_t default_event_destroy(cudaEvent_t event) {
    return cudaEventDestroy(event);
}

cudaError_t default_event_record(cudaEvent_t event, cudaStream_t stream) {
    return cudaEventRecord(event, stream);
}

cudaError_t default_event_synchronize(cudaEvent_t event) {
    return cudaEventSynchronize(event);
}

cudaError_t default_event_elapsed_time(float* milliseconds, cudaEvent_t start,
                                       cudaEvent_t stop) {
    return cudaEventElapsedTime(milliseconds, start, stop);
}

const runner_internal::CudaApi& default_cuda_api() {
    static const runner_internal::CudaApi api{
        default_malloc,
        default_free,
        default_memcpy_h2d,
        default_memcpy_d2h,
        default_peek_at_last_error,
        default_device_synchronize,
        default_event_create,
        default_event_destroy,
        default_event_record,
        default_event_synchronize,
        default_event_elapsed_time,
    };
    return api;
}

class DeviceBuffer {
   public:
    DeviceBuffer(const runner_internal::CudaApi& cuda_api, std::size_t bytes)
        : cuda_api_(cuda_api), pointer_(nullptr) {
        CUDA_CHECK(cuda_api_.malloc_fn(&pointer_, bytes));
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            (void)cuda_api_.free_fn(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* as_float() {
        return static_cast<float*>(pointer_);
    }

   private:
    const runner_internal::CudaApi& cuda_api_;
    void* pointer_;
};

class EventHandle {
   public:
    explicit EventHandle(const runner_internal::CudaApi& cuda_api)
        : cuda_api_(cuda_api), event_(nullptr) {
        CUDA_CHECK(cuda_api_.event_create_fn(&event_));
    }

    ~EventHandle() {
        if (event_ != nullptr) {
            (void)cuda_api_.event_destroy_fn(event_);
        }
    }

    EventHandle(const EventHandle&) = delete;
    EventHandle& operator=(const EventHandle&) = delete;

    cudaEvent_t get() const {
        return event_;
    }

   private:
    const runner_internal::CudaApi& cuda_api_;
    cudaEvent_t event_;
};

struct ValidationSummary {
    ErrorMetrics metrics;
    std::string selected_path;
    bool passed;
};

std::size_t reference_work(Problem problem) {
    const std::size_t m = static_cast<std::size_t>(problem.m);
    const std::size_t n = static_cast<std::size_t>(problem.n);
    const std::size_t k = static_cast<std::size_t>(problem.k);
    return checked_multiply(checked_multiply(m, n, "reference work"), k,
                            "reference work");
}

void require_reference_available(Problem problem) {
    if (reference_work(problem) >= kMaxCpuReferenceWork) {
        throw std::runtime_error("large-problem reference is not available yet");
    }
}

std::size_t matrix_count(int rows, int columns, std::string_view description) {
    return checked_multiply(static_cast<std::size_t>(rows),
                            static_cast<std::size_t>(columns), description);
}

std::size_t matrix_bytes(std::size_t count, std::string_view description) {
    return checked_multiply(count, sizeof(float), description);
}

std::string selected_path_or_kernel_name(const LaunchResult& result,
                                         const KernelDescriptor& kernel) {
    if (result.selected_path == nullptr || result.selected_path[0] == '\0') {
        return kernel.name;
    }
    return result.selected_path;
}

void print_result_line(std::ostream& output, const KernelDescriptor& kernel,
                       std::string_view selected_path, Problem problem, bool passed,
                       const ErrorMetrics& metrics, double latency_ms,
                       double gflops) {
    output << std::fixed << std::setprecision(6) << "kernel=" << kernel.name
           << " path=" << selected_path << " shape=" << problem.m << 'x'
           << problem.n << 'x' << problem.k << " status="
           << (passed ? "PASS" : "FAIL") << " max_abs=" << metrics.max_abs
           << " max_rel=" << metrics.max_rel << " latency_ms=" << latency_ms
           << " gflops=" << gflops << '\n';
}

ValidationSummary run_validation_phase(
    const RunnerOptions& options, const KernelDescriptor& kernel,
    const runner_internal::CudaApi& cuda_api, float* device_a, float* device_b,
    float* device_c, const std::vector<float>& expected,
    std::vector<float>& actual) {
    std::string selected_path = kernel.name;
    for (int warmup = 0; warmup < options.warmup; ++warmup) {
        selected_path = selected_path_or_kernel_name(
            kernel.launch(device_a, device_b, device_c, options.problem, nullptr), kernel);
    }

    selected_path = selected_path_or_kernel_name(
        kernel.launch(device_a, device_b, device_c, options.problem, nullptr), kernel);

    CUDA_CHECK(cuda_api.peek_at_last_error_fn());
    CUDA_CHECK(cuda_api.device_synchronize_fn());
    CUDA_CHECK(cuda_api.memcpy_d2h_fn(actual.data(), device_c,
                                      matrix_bytes(actual.size(), "C copy bytes")));

    const ErrorMetrics metrics = compare(expected.data(), actual.data(), actual.size());
    return ValidationSummary{metrics, selected_path,
                             passes(metrics, kValidationAtol, kValidationRtol)};
}

double compute_gflops(Problem problem, double average_ms) {
    if (average_ms <= 0.0) {
        return 0.0;
    }

    const double operations = 2.0 * static_cast<double>(problem.m) *
                              static_cast<double>(problem.n) *
                              static_cast<double>(problem.k);
    return operations / (average_ms * 1.0e6);
}

double run_benchmark_phase(const RunnerOptions& options, const KernelDescriptor& kernel,
                           const runner_internal::CudaApi& cuda_api, float* device_a,
                           float* device_b, float* device_c,
                           std::string& selected_path) {
    for (int warmup = 0; warmup < options.warmup; ++warmup) {
        selected_path = selected_path_or_kernel_name(
            kernel.launch(device_a, device_b, device_c, options.problem, nullptr), kernel);
    }

    EventHandle start(cuda_api);
    EventHandle stop(cuda_api);

    CUDA_CHECK(cuda_api.event_record_fn(start.get(), nullptr));
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        selected_path = selected_path_or_kernel_name(
            kernel.launch(device_a, device_b, device_c, options.problem, nullptr), kernel);
    }
    CUDA_CHECK(cuda_api.event_record_fn(stop.get(), nullptr));
    CUDA_CHECK(cuda_api.event_synchronize_fn(stop.get()));

    float total_ms = 0.0F;
    CUDA_CHECK(cuda_api.event_elapsed_time_fn(&total_ms, start.get(), stop.get()));
    return static_cast<double>(total_ms) / static_cast<double>(options.iterations);
}

std::string require_value(int argc, const char* const argv[], int& index,
                          const std::string& option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + option);
    }
    ++index;
    return argv[index];
}

template <typename Integer>
Integer parse_integer(const std::string& text, const std::string& option) {
    Integer value{};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec == std::errc::result_out_of_range) {
        throw std::invalid_argument("integer overflow for " + option + ": " + text);
    }
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument("invalid integer for " + option + ": " + text);
    }
    return value;
}

}  // namespace

RunnerOptions parse_arguments(int argc, const char* const argv[]) {
    RunnerOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help") {
            options.help = true;
        } else if (option == "--list") {
            options.list = true;
        } else if (option == "--kernel") {
            options.kernel = require_value(argc, argv, index, option);
        } else if (option == "--m") {
            options.problem.m = parse_integer<int>(require_value(argc, argv, index, option), option);
        } else if (option == "--n") {
            options.problem.n = parse_integer<int>(require_value(argc, argv, index, option), option);
        } else if (option == "--k") {
            options.problem.k = parse_integer<int>(require_value(argc, argv, index, option), option);
        } else if (option == "--mode") {
            const std::string value = require_value(argc, argv, index, option);
            if (value == "validate") {
                options.mode = RunMode::validate;
            } else if (value == "benchmark") {
                options.mode = RunMode::benchmark;
            } else {
                throw std::invalid_argument(
                    "invalid mode: " + value + " (expected validate or benchmark)");
            }
        } else if (option == "--warmup") {
            options.warmup = parse_integer<int>(require_value(argc, argv, index, option), option);
        } else if (option == "--iterations") {
            options.iterations = parse_integer<int>(require_value(argc, argv, index, option), option);
        } else if (option == "--seed") {
            options.seed = parse_integer<std::uint32_t>(
                require_value(argc, argv, index, option), option);
        } else if (option == "--csv") {
            options.csv_path = require_value(argc, argv, index, option);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    return options;
}

void validate_options(const RunnerOptions& options) {
    if (options.help || options.list) {
        return;
    }
    if (options.kernel.empty()) {
        throw std::invalid_argument("normal run requires --kernel");
    }
    if (options.problem.m <= 0) {
        throw std::invalid_argument("--m must be a positive integer");
    }
    if (options.problem.n <= 0) {
        throw std::invalid_argument("--n must be a positive integer");
    }
    if (options.problem.k <= 0) {
        throw std::invalid_argument("--k must be a positive integer");
    }
    if (options.warmup < 0) {
        throw std::invalid_argument("--warmup must be a nonnegative integer");
    }
    if (options.iterations <= 0) {
        throw std::invalid_argument("--iterations must be a positive integer");
    }
    if (!options.csv_path.empty()) {
        throw std::invalid_argument("CSV output is not implemented yet");
    }
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             std::string_view description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error("size overflow while computing " +
                                  std::string(description));
    }
    return left * right;
}

std::vector<float> generate_input(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    std::vector<float> values(count);
    for (float& value : values) {
        value = distribution(generator);
    }
    return values;
}

int run(const RunnerOptions& options, std::ostream& output) {
    validate_options(options);

    if (options.help) {
        output
            << "Usage: gemm_runner [options]\n"
            << "  --help                  Show this help message\n"
            << "  --list                  List registered kernels and exit\n"
            << "  --kernel <name>         Kernel name to run\n"
            << "  --m <int>               Positive M dimension\n"
            << "  --n <int>               Positive N dimension\n"
            << "  --k <int>               Positive K dimension\n"
            << "  --mode <validate|benchmark>  Default: validate\n"
            << "  --warmup <count>       Default: 5\n"
            << "  --iterations <count>   Default: 20\n"
            << "  --seed <uint>          Default: 1234\n"
            << "  --csv <path>           Parsed now; currently returns not implemented\n"
            << "Validate mode reports latency_ms=0 and gflops=0 until timing is enabled.\n";
        return 0;
    }

    if (options.list) {
        const std::vector<KernelDescriptor> kernels = registered_kernels();
        if (kernels.empty()) {
            output << "No kernels registered.\n";
            return 0;
        }
        for (const KernelDescriptor& kernel : kernels) {
            output << kernel.name << '\n';
        }
        return 0;
    }

    const KernelDescriptor* kernel = find_kernel(options.kernel);
    if (kernel == nullptr) {
        throw std::invalid_argument("unknown kernel: " + options.kernel);
    }

    return runner_internal::run_with_kernel(options, *kernel, output);
}

namespace runner_internal {

int run_with_kernel(const RunnerOptions& options, const KernelDescriptor& kernel,
                    std::ostream& output) {
    return run_with_kernel(options, kernel, output, default_cuda_api());
}

int run_with_kernel(const RunnerOptions& options, const KernelDescriptor& kernel,
                    std::ostream& output, const CudaApi& cuda_api) {
    // This direct test/injection boundary may bypass run(), so validate again here.
    validate_options(options);
    require_reference_available(options.problem);

    const std::size_t a_count = matrix_count(options.problem.m, options.problem.k,
                                             "A elements");
    const std::size_t b_count = matrix_count(options.problem.k, options.problem.n,
                                             "B elements");
    const std::size_t c_count = matrix_count(options.problem.m, options.problem.n,
                                             "C elements");

    std::vector<float> host_a = generate_input(a_count, options.seed);
    std::vector<float> host_b = generate_input(b_count, options.seed + 1U);
    std::vector<float> expected(c_count, 0.0F);
    std::vector<float> actual(c_count, 0.0F);

    reference_cpu(host_a.data(), host_b.data(), expected.data(), options.problem.m,
                  options.problem.n, options.problem.k);

    DeviceBuffer device_a(cuda_api, matrix_bytes(a_count, "A bytes"));
    DeviceBuffer device_b(cuda_api, matrix_bytes(b_count, "B bytes"));
    DeviceBuffer device_c(cuda_api, matrix_bytes(c_count, "C bytes"));

    CUDA_CHECK(cuda_api.memcpy_h2d_fn(device_a.as_float(), host_a.data(),
                                      matrix_bytes(a_count, "A bytes")));
    CUDA_CHECK(cuda_api.memcpy_h2d_fn(device_b.as_float(), host_b.data(),
                                      matrix_bytes(b_count, "B bytes")));

    ValidationSummary validation =
        run_validation_phase(options, kernel, cuda_api, device_a.as_float(),
                             device_b.as_float(), device_c.as_float(), expected, actual);

    if (options.mode == RunMode::validate) {
        print_result_line(output, kernel, validation.selected_path, options.problem,
                          validation.passed, validation.metrics, 0.0, 0.0);
        return validation.passed ? 0 : 1;
    }

    if (!validation.passed) {
        print_result_line(output, kernel, validation.selected_path, options.problem,
                          false, validation.metrics, 0.0, 0.0);
        return 1;
    }

    std::string selected_path = validation.selected_path;
    const double average_ms = run_benchmark_phase(
        options, kernel, cuda_api, device_a.as_float(), device_b.as_float(),
        device_c.as_float(), selected_path);
    print_result_line(output, kernel, selected_path, options.problem, true,
                      validation.metrics, average_ms,
                      compute_gflops(options.problem, average_ms));
    return 0;
}

}  // namespace runner_internal

}  // namespace gemm
