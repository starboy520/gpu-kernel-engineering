#include "gemm/runner.hpp"

#include "gemm/kernel.hpp"
#include "gemm/reference.hpp"
#include "gemm/validation.hpp"
#include "gpu_kernel/cuda_check.hpp"
#include "gpu_kernel/runner_utils.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace gemm {
namespace {

constexpr double kValidationAtol = 1.0e-3;
constexpr double kValidationRtol = 1.0e-3;
constexpr double kLargeReferenceAtol = 1.0e-3;
constexpr double kLargeReferenceRtol = 2.0e-3;
constexpr char kCsvHeader[] =
    "timestamp,git_commit,gpu,cuda,nvcc,kernel,path,m,n,k,warmup,iterations,"
    "latency_ms,gflops,passed,max_abs,max_rel,reference";

struct CsvMetadata {
    std::string timestamp;
    std::string git_commit;
    std::string gpu;
    std::string cuda;
    std::string nvcc;
};

cudaError_t default_malloc(void **pointer, std::size_t bytes) {
    return cudaMalloc(pointer, bytes);
}

cudaError_t default_free(void *pointer) { return cudaFree(pointer); }

cudaError_t default_memcpy_h2d(void *destination, const void *source,
                               std::size_t bytes) {
    return cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);
}

cudaError_t default_memcpy_d2h(void *destination, const void *source,
                               std::size_t bytes) {
    return cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);
}

cudaError_t default_peek_at_last_error() { return cudaPeekAtLastError(); }

cudaError_t default_device_synchronize() { return cudaDeviceSynchronize(); }

cudaError_t default_event_create(cudaEvent_t *event) {
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

cudaError_t default_event_elapsed_time(float *milliseconds, cudaEvent_t start,
                                       cudaEvent_t stop) {
    return cudaEventElapsedTime(milliseconds, start, stop);
}

const runner_internal::CudaApi &default_cuda_api() {
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
    DeviceBuffer(const runner_internal::CudaApi &cuda_api, std::size_t bytes)
        : cuda_api_(cuda_api), pointer_(nullptr) {
        GPU_CUDA_CHECK(cuda_api_.malloc_fn(&pointer_, bytes));
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            (void)cuda_api_.free_fn(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    float *as_float() { return static_cast<float *>(pointer_); }

  private:
    const runner_internal::CudaApi &cuda_api_;
    void *pointer_;
};

class EventHandle {
  public:
    explicit EventHandle(const runner_internal::CudaApi &cuda_api)
        : cuda_api_(cuda_api), event_(nullptr) {
        GPU_CUDA_CHECK(cuda_api_.event_create_fn(&event_));
    }

    ~EventHandle() {
        if (event_ != nullptr) {
            (void)cuda_api_.event_destroy_fn(event_);
        }
    }

    EventHandle(const EventHandle &) = delete;
    EventHandle &operator=(const EventHandle &) = delete;

    cudaEvent_t get() const { return event_; }

  private:
    const runner_internal::CudaApi &cuda_api_;
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

void require_reference_available(
    Problem problem, runner_internal::LargeReferenceFn large_reference) {
    if (reference_work(problem) >= kMaxCpuReferenceWork &&
        large_reference == nullptr) {
        throw std::runtime_error(
            "large-problem reference is not available yet");
    }
}

std::size_t matrix_count(int rows, int columns, std::string_view description) {
    return checked_multiply(static_cast<std::size_t>(rows),
                            static_cast<std::size_t>(columns), description);
}

std::size_t matrix_bytes(std::size_t count, std::string_view description) {
    return checked_multiply(count, sizeof(float), description);
}

std::string selected_path_or_kernel_name(const LaunchResult &result,
                                         const KernelDescriptor &kernel) {
    if (result.selected_path == nullptr || result.selected_path[0] == '\0') {
        return kernel.name;
    }
    return result.selected_path;
}

std::string format_double(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    return stream.str();
}

std::string csv_escape(std::string_view field) {
    if (field.find_first_of(",\"\n\r") == std::string_view::npos) {
        return std::string(field);
    }

    std::string escaped;
    escaped.reserve(field.size() + 2U);
    escaped.push_back('"');
    for (char character : field) {
        if (character == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

std::string read_env_value(const char *primary_name,
                           const char *fallback_name = nullptr) {
    if (const char *value = std::getenv(primary_name); value != nullptr) {
        return value;
    }
    if (fallback_name != nullptr) {
        if (const char *value = std::getenv(fallback_name); value != nullptr) {
            return value;
        }
    }
    return "";
}

CsvMetadata load_csv_metadata() {
    return CsvMetadata{
        read_env_value("GEMM_BENCH_TIMESTAMP"),
        read_env_value("GEMM_GIT_COMMIT", "GIT_COMMIT"),
        read_env_value("GEMM_GPU", "GPU"),
        read_env_value("GEMM_CUDA", "CUDA"),
        read_env_value("GEMM_NVCC", "NVCC"),
    };
}

bool csv_file_needs_header(const std::string &csv_path) {
    std::ifstream input(csv_path);
    if (!input.is_open()) {
        return true;
    }

    std::string header;
    if (!std::getline(input, header)) {
        return true;
    }

    if (header == kCsvHeader) {
        return false;
    }

    throw std::runtime_error("existing CSV header mismatch: " + csv_path);
}

void append_csv_row(const RunnerOptions &options,
                    const KernelDescriptor &kernel,
                    std::string_view selected_path, bool passed,
                    const ErrorMetrics &metrics, double latency_ms,
                    double gflops, std::string_view reference_source) {
    if (options.csv_path.empty()) {
        return;
    }

    const CsvMetadata metadata = load_csv_metadata();
    const bool needs_header = csv_file_needs_header(options.csv_path);
    std::ofstream output(options.csv_path, std::ios::app);
    if (!output.is_open()) {
        throw std::runtime_error("failed to open CSV output: " +
                                 options.csv_path);
    }

    if (needs_header) {
        output << kCsvHeader << '\n';
    }

    output << csv_escape(metadata.timestamp) << ','
           << csv_escape(metadata.git_commit) << ',' << csv_escape(metadata.gpu)
           << ',' << csv_escape(metadata.cuda) << ','
           << csv_escape(metadata.nvcc) << ',' << csv_escape(kernel.name) << ','
           << csv_escape(selected_path) << ',' << options.problem.m << ','
           << options.problem.n << ',' << options.problem.k << ','
           << options.warmup << ',' << options.iterations << ','
           << format_double(latency_ms) << ',' << format_double(gflops) << ','
           << (passed ? "true" : "false") << ','
           << format_double(metrics.max_abs) << ','
           << format_double(metrics.max_rel) << ','
           << csv_escape(reference_source) << '\n';

    if (!output) {
        throw std::runtime_error("failed to write CSV row: " +
                                 options.csv_path);
    }
}

void print_result_line(std::ostream &output, const KernelDescriptor &kernel,
                       std::string_view selected_path, Problem problem,
                       bool passed, const ErrorMetrics &metrics,
                       double latency_ms, double gflops,
                       std::string_view reference_source) {
    output << std::fixed << std::setprecision(6) << "kernel=" << kernel.name
           << " path=" << selected_path << " shape=" << problem.m << 'x'
           << problem.n << 'x' << problem.k
           << " status=" << (passed ? "PASS" : "FAIL")
           << " max_abs=" << metrics.max_abs << " max_rel=" << metrics.max_rel
           << " latency_ms=" << latency_ms << " gflops=" << gflops
           << " reference=" << reference_source << '\n';
}

ValidationSummary run_validation_phase(
    const RunnerOptions &options, const KernelDescriptor &kernel,
    const runner_internal::CudaApi &cuda_api, float *device_a, float *device_b,
    float *device_c, const std::vector<float> &expected,
    std::vector<float> &actual, double atol, double rtol) {
    std::string selected_path = kernel.name;
    for (int warmup = 0; warmup < options.warmup; ++warmup) {
        selected_path = selected_path_or_kernel_name(
            kernel.launch(device_a, device_b, device_c, options.problem,
                          nullptr),
            kernel);
    }

    selected_path = selected_path_or_kernel_name(
        kernel.launch(device_a, device_b, device_c, options.problem, nullptr),
        kernel);

    GPU_CUDA_CHECK(cuda_api.peek_at_last_error_fn());
    GPU_CUDA_CHECK(cuda_api.device_synchronize_fn());
    GPU_CUDA_CHECK(cuda_api.memcpy_d2h_fn(
        actual.data(), device_c, matrix_bytes(actual.size(), "C copy bytes")));

    const ErrorMetrics metrics =
        compare(expected.data(), actual.data(), actual.size());
    return ValidationSummary{metrics, selected_path,
                             passes(metrics, atol, rtol)};
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

double run_benchmark_phase(const RunnerOptions &options,
                           const KernelDescriptor &kernel,
                           const runner_internal::CudaApi &cuda_api,
                           float *device_a, float *device_b, float *device_c,
                           std::string &selected_path) {
    for (int warmup = 0; warmup < options.warmup; ++warmup) {
        selected_path = selected_path_or_kernel_name(
            kernel.launch(device_a, device_b, device_c, options.problem,
                          nullptr),
            kernel);
    }

    EventHandle start(cuda_api);
    EventHandle stop(cuda_api);

    GPU_CUDA_CHECK(cuda_api.event_record_fn(start.get(), nullptr));
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        selected_path = selected_path_or_kernel_name(
            kernel.launch(device_a, device_b, device_c, options.problem,
                          nullptr),
            kernel);
    }
    GPU_CUDA_CHECK(cuda_api.event_record_fn(stop.get(), nullptr));
    GPU_CUDA_CHECK(cuda_api.event_synchronize_fn(stop.get()));

    float total_ms = 0.0F;
    GPU_CUDA_CHECK(
        cuda_api.event_elapsed_time_fn(&total_ms, start.get(), stop.get()));
    return static_cast<double>(total_ms) /
           static_cast<double>(options.iterations);
}

} // namespace

RunnerOptions parse_arguments(int argc, const char *const argv[]) {
    RunnerOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help") {
            options.help = true;
        } else if (option == "--list") {
            options.list = true;
        } else if (option == "--kernel") {
            options.kernel =
                gpu_kernel::require_value(argc, argv, index, option);
        } else if (option == "--m") {
            options.problem.m = gpu_kernel::parse_integer<int>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--n") {
            options.problem.n = gpu_kernel::parse_integer<int>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--k") {
            options.problem.k = gpu_kernel::parse_integer<int>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--mode") {
            const std::string value =
                gpu_kernel::require_value(argc, argv, index, option);
            if (value == "validate") {
                options.mode = RunMode::validate;
            } else if (value == "benchmark") {
                options.mode = RunMode::benchmark;
            } else {
                throw std::invalid_argument(
                    "invalid mode: " + value +
                    " (expected validate or benchmark)");
            }
        } else if (option == "--warmup") {
            options.warmup = gpu_kernel::parse_integer<int>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--iterations") {
            options.iterations = gpu_kernel::parse_integer<int>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--seed") {
            options.seed = gpu_kernel::parse_integer<std::uint32_t>(
                gpu_kernel::require_value(argc, argv, index, option), option);
        } else if (option == "--csv") {
            options.csv_path =
                gpu_kernel::require_value(argc, argv, index, option);
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
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             std::string_view description) {
    return gpu_kernel::checked_multiply(left, right, description);
}

std::vector<float> generate_input(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    std::vector<float> values(count);
    for (float &value : values) {
        value = distribution(generator);
    }
    return values;
}

int run(const RunnerOptions &options, std::ostream &output) {
    validate_options(options);

    if (options.help) {
        output << "Usage: gemm_runner [options]\n"
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
               << "  --csv <path>           Append machine-readable benchmark "
                  "rows to CSV\n"
               << "Validate mode reports latency_ms=0 and gflops=0 until "
                  "timing is enabled.\n";
        return 0;
    }

    if (options.list) {
        const std::vector<KernelDescriptor> kernels = registered_kernels();
        if (kernels.empty()) {
            output << "No kernels registered.\n";
            return 0;
        }
        for (const KernelDescriptor &kernel : kernels) {
            output << kernel.name << '\n';
        }
        return 0;
    }

    const KernelDescriptor *kernel = find_kernel(options.kernel);
    if (kernel == nullptr) {
        throw std::invalid_argument("unknown kernel: " + options.kernel);
    }

    return runner_internal::run_with_kernel(options, *kernel, output);
}

namespace runner_internal {

bool uses_cpu_reference(Problem problem) {
    return reference_work(problem) < kMaxCpuReferenceWork;
}

ValidationTolerances validation_tolerances(bool cpu_reference) {
    return cpu_reference
               ? ValidationTolerances{kValidationAtol, kValidationRtol}
               : ValidationTolerances{kLargeReferenceAtol, kLargeReferenceRtol};
}

int run_with_kernel(const RunnerOptions &options,
                    const KernelDescriptor &kernel, std::ostream &output) {
    return run_with_kernel(options, kernel, output, default_cuda_api(),
                           reference_cublas_device);
}

int run_with_kernel(const RunnerOptions &options,
                    const KernelDescriptor &kernel, std::ostream &output,
                    const CudaApi &cuda_api) {
    return run_with_kernel(options, kernel, output, cuda_api, nullptr);
}

int run_with_kernel(const RunnerOptions &options,
                    const KernelDescriptor &kernel, std::ostream &output,
                    const CudaApi &cuda_api, LargeReferenceFn large_reference) {
    // This direct test/injection boundary may bypass run(), so validate again
    // here.
    validate_options(options);
    require_reference_available(options.problem, large_reference);
    const bool use_cpu_reference = uses_cpu_reference(options.problem);

    const std::size_t a_count =
        matrix_count(options.problem.m, options.problem.k, "A elements");
    const std::size_t b_count =
        matrix_count(options.problem.k, options.problem.n, "B elements");
    const std::size_t c_count =
        matrix_count(options.problem.m, options.problem.n, "C elements");

    std::vector<float> host_a = generate_input(a_count, options.seed);
    std::vector<float> host_b = generate_input(b_count, options.seed + 1U);
    std::vector<float> expected(c_count, 0.0F);
    std::vector<float> actual(c_count, 0.0F);

    if (use_cpu_reference) {
        reference_cpu(host_a.data(), host_b.data(), expected.data(),
                      options.problem.m, options.problem.n, options.problem.k);
    }

    DeviceBuffer device_a(cuda_api, matrix_bytes(a_count, "A bytes"));
    DeviceBuffer device_b(cuda_api, matrix_bytes(b_count, "B bytes"));
    DeviceBuffer device_c(cuda_api, matrix_bytes(c_count, "C bytes"));

    GPU_CUDA_CHECK(cuda_api.memcpy_h2d_fn(device_a.as_float(), host_a.data(),
                                          matrix_bytes(a_count, "A bytes")));
    GPU_CUDA_CHECK(cuda_api.memcpy_h2d_fn(device_b.as_float(), host_b.data(),
                                          matrix_bytes(b_count, "B bytes")));

    if (!use_cpu_reference) {
        large_reference(device_a.as_float(), device_b.as_float(),
                        device_c.as_float(), options.problem, nullptr);
        GPU_CUDA_CHECK(cuda_api.peek_at_last_error_fn());
        GPU_CUDA_CHECK(cuda_api.device_synchronize_fn());
        GPU_CUDA_CHECK(cuda_api.memcpy_d2h_fn(
            expected.data(), device_c.as_float(),
            matrix_bytes(expected.size(), "reference C copy bytes")));
    }

    const std::string_view reference_source =
        use_cpu_reference ? "cpu" : "cublas-pedantic-fp32";
    const ValidationTolerances tolerances =
        validation_tolerances(use_cpu_reference);

    ValidationSummary validation =
        run_validation_phase(options, kernel, cuda_api, device_a.as_float(),
                             device_b.as_float(), device_c.as_float(), expected,
                             actual, tolerances.atol, tolerances.rtol);

    if (options.mode == RunMode::validate) {
        print_result_line(output, kernel, validation.selected_path,
                          options.problem, validation.passed,
                          validation.metrics, 0.0, 0.0, reference_source);
        append_csv_row(options, kernel, validation.selected_path,
                       validation.passed, validation.metrics, 0.0, 0.0,
                       reference_source);
        return validation.passed ? 0 : 1;
    }

    if (!validation.passed) {
        print_result_line(output, kernel, validation.selected_path,
                          options.problem, false, validation.metrics, 0.0, 0.0,
                          reference_source);
        append_csv_row(options, kernel, validation.selected_path, false,
                       validation.metrics, 0.0, 0.0, reference_source);
        return 1;
    }

    std::string selected_path = validation.selected_path;
    const double average_ms = run_benchmark_phase(
        options, kernel, cuda_api, device_a.as_float(), device_b.as_float(),
        device_c.as_float(), selected_path);
    const double gflops = compute_gflops(options.problem, average_ms);
    print_result_line(output, kernel, selected_path, options.problem, true,
                      validation.metrics, average_ms, gflops, reference_source);
    append_csv_row(options, kernel, selected_path, true, validation.metrics,
                   average_ms, gflops, reference_source);
    return 0;
}

} // namespace runner_internal

} // namespace gemm
