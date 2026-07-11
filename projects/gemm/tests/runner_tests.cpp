#include "gemm/cuda_check.hpp"
#include "gemm/kernel.hpp"
#include "gemm/reference.hpp"
#include "gemm/runner.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
int cuda_evaluations = 0;
int cublas_evaluations = 0;
int large_reference_evaluations = 0;
int scripted_d2h_evaluations = 0;
float scripted_elapsed_ms = 0.0F;
int temp_csv_counter = 0;

struct FakeKernelState {
    int launches = 0;
    float delta = 0.0F;
    const char* path = "fake_path";
};

struct AllocationTracker {
    int malloc_calls = 0;
};

FakeKernelState* active_fake_kernel_state = nullptr;
AllocationTracker* active_allocation_tracker = nullptr;

template <typename T>
class ScopedBinding {
   public:
    ScopedBinding(T*& target, T* replacement)
        : target_(target), previous_(target) {
        target_ = replacement;
    }

    ~ScopedBinding() {
        target_ = previous_;
    }

    ScopedBinding(const ScopedBinding&) = delete;
    ScopedBinding& operator=(const ScopedBinding&) = delete;

   private:
    T*& target_;
    T* previous_;
};

class ScopedEnvironmentVariable {
   public:
    ScopedEnvironmentVariable(const char* name, const char* value)
        : name_(name), had_previous_(false) {
        const char* previous = std::getenv(name_);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_value_ = previous;
        }

        if (value != nullptr) {
            setenv(name_, value, 1);
        } else {
            unsetenv(name_);
        }
    }

    ~ScopedEnvironmentVariable() {
        if (had_previous_) {
            setenv(name_, previous_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
    ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

   private:
    const char* name_;
    bool had_previous_;
    std::string previous_value_;
};

gemm::LaunchResult fake_reference_launcher(const float* a, const float* b, float* c,
                                          gemm::Problem problem, cudaStream_t) {
    if (active_fake_kernel_state == nullptr) {
        throw std::runtime_error("fake kernel state must be configured");
    }

    ++active_fake_kernel_state->launches;

    const std::size_t a_count = static_cast<std::size_t>(problem.m) *
                                static_cast<std::size_t>(problem.k);
    const std::size_t b_count = static_cast<std::size_t>(problem.k) *
                                static_cast<std::size_t>(problem.n);
    const std::size_t c_count = static_cast<std::size_t>(problem.m) *
                                static_cast<std::size_t>(problem.n);

    std::vector<float> host_a(a_count);
    std::vector<float> host_b(b_count);
    std::vector<float> host_c(c_count);

    CUDA_CHECK(cudaMemcpy(host_a.data(), a, sizeof(float) * a_count,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_b.data(), b, sizeof(float) * b_count,
                          cudaMemcpyDeviceToHost));

    gemm::reference_cpu(host_a.data(), host_b.data(), host_c.data(), problem.m,
                        problem.n, problem.k);
    for (float& value : host_c) {
        value += active_fake_kernel_state->delta;
    }

    CUDA_CHECK(cudaMemcpy(c, host_c.data(), sizeof(float) * c_count,
                          cudaMemcpyHostToDevice));
    return {active_fake_kernel_state->path, false};
}

cudaError_t counting_malloc(void**, std::size_t) {
    if (active_allocation_tracker != nullptr) {
        ++active_allocation_tracker->malloc_calls;
    }
    return cudaSuccess;
}

cudaError_t counting_free(void*) {
    return cudaSuccess;
}

cudaError_t counting_memcpy_h2d(void*, const void*, std::size_t) {
    return cudaSuccess;
}

cudaError_t counting_memcpy_d2h(void*, const void*, std::size_t) {
    return cudaSuccess;
}

cudaError_t scripted_memcpy_d2h(void* destination, const void*, std::size_t bytes) {
    const float value = scripted_d2h_evaluations++ == 0 ? 1.0F : 1.01F;
    float* output = static_cast<float*>(destination);
    std::fill(output, output + bytes / sizeof(float), value);
    return cudaSuccess;
}

cudaError_t counting_peek_at_last_error() {
    return cudaSuccess;
}

cudaError_t counting_device_synchronize() {
    return cudaSuccess;
}

cudaError_t counting_event_create(cudaEvent_t*) {
    return cudaSuccess;
}

cudaError_t counting_event_destroy(cudaEvent_t) {
    return cudaSuccess;
}

cudaError_t counting_event_record(cudaEvent_t, cudaStream_t) {
    return cudaSuccess;
}

cudaError_t counting_event_synchronize(cudaEvent_t) {
    return cudaSuccess;
}

cudaError_t counting_event_elapsed_time(float* milliseconds, cudaEvent_t,
                                        cudaEvent_t) {
    if (milliseconds != nullptr) {
        *milliseconds = 0.0F;
    }
    return cudaSuccess;
}

cudaError_t scripted_event_elapsed_time(float* milliseconds, cudaEvent_t,
                                        cudaEvent_t) {
    if (milliseconds != nullptr) {
        *milliseconds = scripted_elapsed_ms;
    }
    return cudaSuccess;
}

cudaError_t host_malloc(void** pointer, std::size_t bytes) {
    *pointer = std::malloc(bytes);
    return *pointer == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
}

cudaError_t host_free(void* pointer) {
    std::free(pointer);
    return cudaSuccess;
}

cudaError_t host_memcpy(void* destination, const void* source, std::size_t bytes) {
    std::memcpy(destination, source, bytes);
    return cudaSuccess;
}

void fake_large_reference(const float*, const float*, float* c,
                          gemm::Problem problem, cudaStream_t) {
    ++large_reference_evaluations;
    const std::size_t count = static_cast<std::size_t>(problem.m) *
                              static_cast<std::size_t>(problem.n);
    std::fill(c, c + count, 0.0F);
}

void fake_counting_large_reference(const float*, const float*, float*,
                                   gemm::Problem, cudaStream_t) {
    ++large_reference_evaluations;
}

gemm::LaunchResult fake_zero_launcher(const float*, const float*, float* c,
                                      gemm::Problem problem, cudaStream_t) {
    const std::size_t count = static_cast<std::size_t>(problem.m) *
                              static_cast<std::size_t>(problem.n);
    std::fill(c, c + count, 0.0F);
    return {"fake_zero", false};
}

gemm::LaunchResult fake_biased_launcher(const float*, const float*, float* c,
                                        gemm::Problem problem, cudaStream_t) {
    (void)c;
    (void)problem;
    return {"fake_biased", false};
}

std::filesystem::path make_temp_csv_path() {
    return std::filesystem::temp_directory_path() /
           ("gemm_runner_csv_test_" + std::to_string(temp_csv_counter++) + ".csv");
}

std::vector<std::string> read_lines(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }
    return lines;
}

std::vector<std::string> split_csv_row(const std::string& row) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (std::size_t index = 0; index < row.size(); ++index) {
        const char character = row[index];
        if (character == '"') {
            if (in_quotes && index + 1 < row.size() && row[index + 1] == '"') {
                field.push_back('"');
                ++index;
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }

        if (character == ',' && !in_quotes) {
            fields.push_back(field);
            field.clear();
            continue;
        }

        field.push_back(character);
    }

    fields.push_back(field);
    return fields;
}

gemm::runner_internal::CudaApi make_host_cuda_api(
    cudaError_t (*memcpy_d2h_fn)(void*, const void*, std::size_t),
    cudaError_t (*event_elapsed_time_fn)(float*, cudaEvent_t, cudaEvent_t)) {
    return gemm::runner_internal::CudaApi{
        host_malloc,
        host_free,
        host_memcpy,
        memcpy_d2h_fn,
        counting_peek_at_last_error,
        counting_device_synchronize,
        counting_event_create,
        counting_event_destroy,
        counting_event_record,
        counting_event_synchronize,
        event_elapsed_time_fn,
    };
}

}  // namespace

namespace {

namespace runner_internal = gemm::runner_internal;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Function>
void check_throws(Function&& function,
                  const std::string& expected_message,
                  const std::string& description) {
    try {
        function();
        check(false, description + " did not throw");
    } catch (const std::exception& error) {
        check(error.what() == expected_message,
              description + " message was: " + error.what());
    }
}

gemm::RunnerOptions parse(std::initializer_list<const char*> arguments) {
    const std::vector<const char*> argv(arguments);
    return gemm::parse_arguments(static_cast<int>(argv.size()), argv.data());
}

gemm::RunnerOptions make_run_options(gemm::RunMode mode, int warmup, int iterations,
                                     int m, int n, int k) {
    gemm::RunnerOptions options;
    options.kernel = "fake_reference";
    options.problem = gemm::Problem{m, n, k};
    options.mode = mode;
    options.warmup = warmup;
    options.iterations = iterations;
    options.seed = 7U;
    return options;
}

cudaError_t failing_cuda_call() {
    ++cuda_evaluations;
    return cudaErrorInvalidValue;
}

cublasStatus_t failing_cublas_call() {
    ++cublas_evaluations;
    return CUBLAS_STATUS_INVALID_VALUE;
}

void test_error_macros_throw_readable_single_evaluation_errors() {
    try {
        CUDA_CHECK(failing_cuda_call());
        check(false, "CUDA_CHECK did not throw");
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        check(message.find("failing_cuda_call()") != std::string::npos,
              "CUDA_CHECK includes expression");
        check(message.find("cudaErrorInvalidValue") != std::string::npos,
              "CUDA_CHECK includes readable status");
        check(message.find("runner_tests.cpp:") != std::string::npos,
              "CUDA_CHECK includes source location");
    }
    check(cuda_evaluations == 1, "CUDA_CHECK evaluates expression once");

    try {
        CUBLAS_CHECK(failing_cublas_call());
        check(false, "CUBLAS_CHECK did not throw");
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        check(message.find("failing_cublas_call()") != std::string::npos,
              "CUBLAS_CHECK includes expression");
        check(message.find("CUBLAS_STATUS_INVALID_VALUE") != std::string::npos,
              "CUBLAS_CHECK includes readable status");
        check(message.find("runner_tests.cpp:") != std::string::npos,
              "CUBLAS_CHECK includes source location");
    }
    check(cublas_evaluations == 1, "CUBLAS_CHECK evaluates expression once");
}

void test_parser_defaults_and_full_command() {
    const gemm::RunnerOptions defaults = parse({"gemm_runner"});
    check(defaults.mode == gemm::RunMode::validate, "default mode is validate");
    check(defaults.warmup == 5, "default warmup is 5");
    check(defaults.iterations == 20, "default iterations is 20");
    check(defaults.seed == 1234U, "default seed is 1234");

    const gemm::RunnerOptions options = parse(
        {"gemm_runner", "--kernel", "tiled", "--m", "17", "--n", "19",
         "--k", "23", "--mode", "benchmark", "--warmup", "0",
         "--iterations", "7", "--seed", "4294967295", "--csv", "out.csv"});
    check(options.kernel == "tiled", "parser stores kernel");
    check(options.problem.m == 17 && options.problem.n == 19 && options.problem.k == 23,
          "parser stores dimensions");
    check(options.mode == gemm::RunMode::benchmark, "parser stores benchmark mode");
    check(options.warmup == 0 && options.iterations == 7, "parser stores counts");
    check(options.seed == std::numeric_limits<std::uint32_t>::max(),
          "parser accepts maximum unsigned seed");
    check(options.csv_path == "out.csv", "parser stores CSV path");
}

void test_parser_rejects_bad_tokens() {
    check_throws([] { parse({"gemm_runner", "--unknown"}); },
                 "unknown option: --unknown", "unknown option");
    check_throws([] { parse({"gemm_runner", "--kernel"}); },
                 "missing value for --kernel", "missing value");
    check_throws([] { parse({"gemm_runner", "--m", "abc"}); },
                 "invalid integer for --m: abc", "non-numeric integer");
    check_throws([] { parse({"gemm_runner", "--m", "2147483648"}); },
                 "integer overflow for --m: 2147483648", "signed overflow");
    check_throws([] { parse({"gemm_runner", "--seed", "4294967296"}); },
                 "integer overflow for --seed: 4294967296", "unsigned overflow");
    check_throws([] { parse({"gemm_runner", "--mode", "fast"}); },
                 "invalid mode: fast (expected validate or benchmark)", "invalid mode");
}

void test_option_validation() {
    gemm::validate_options(parse({"gemm_runner", "--list"}));
    gemm::validate_options(parse({"gemm_runner", "--help"}));
    gemm::validate_options(parse({"gemm_runner", "--kernel", "naive", "--m", "1",
                                  "--n", "2", "--k", "3"}));
    gemm::validate_options(parse({"gemm_runner", "--kernel", "naive", "--m", "1",
                                  "--n", "2", "--k", "3", "--csv", "out.csv"}));

    check_throws([] { gemm::validate_options(parse({"gemm_runner"})); },
                 "normal run requires --kernel", "missing kernel");
    check_throws(
        [] {
            gemm::validate_options(parse({"gemm_runner", "--kernel", "naive",
                                          "--m", "0", "--n", "2", "--k", "3"}));
        },
        "--m must be a positive integer", "zero M");
    check_throws(
        [] {
            gemm::validate_options(parse({"gemm_runner", "--kernel", "naive",
                                          "--m", "1", "--n", "0", "--k", "3",
                                          "--csv", "out.csv"}));
        },
        "--n must be a positive integer", "CSV path still validates dimensions");
}

void test_checked_multiply() {
    check(gemm::checked_multiply(7, 9, "matrix elements") == 63,
          "checked_multiply computes product");
    check(gemm::checked_multiply(0, std::numeric_limits<std::size_t>::max(),
                                 "matrix elements") == 0,
          "checked_multiply handles zero");
    check_throws(
        [] {
            gemm::checked_multiply(std::numeric_limits<std::size_t>::max(), 2,
                                   "matrix bytes");
        },
        "size overflow while computing matrix bytes", "checked multiplication overflow");
}

void test_input_generation_is_deterministic_and_bounded() {
    const std::vector<float> first = gemm::generate_input(1024, 42U);
    const std::vector<float> second = gemm::generate_input(1024, 42U);
    const std::vector<float> different = gemm::generate_input(1024, 43U);

    check(first == second, "input generation is deterministic for a seed");
    check(first != different, "input generation varies by seed");
    for (float value : first) {
        check(value >= -0.5F && value <= 0.5F, "input value is in [-0.5, 0.5]");
    }
}

void test_registry_contains_author_kernels_then_cublas_baseline() {
    const std::vector<gemm::KernelDescriptor> kernels = gemm::registered_kernels();
    check(kernels.size() == 6, "registry contains exactly six kernels");
    if (kernels.size() == 6) {
          check(std::string_view(kernels[0].name) == "naive",
              "first registered kernel is naive");
          check(kernels[0].launch == gemm::launch_naive,
              "naive descriptor uses launch_naive");
          check(kernels[0].author_kernel, "naive is marked as an author kernel");
          check(std::string_view(kernels[1].name) == "shared",
              "second registered kernel is shared");
          check(kernels[1].launch == gemm::launch_shared_tiled,
              "shared descriptor uses launch_shared_tiled");
          check(kernels[1].author_kernel, "shared is marked as an author kernel");
          check(std::string_view(kernels[2].name) == "register",
              "third registered kernel is register");
          check(kernels[2].launch == gemm::launch_register_tiled,
              "register descriptor uses launch_register_tiled");
          check(kernels[2].author_kernel, "register is marked as an author kernel");
          check(std::string_view(kernels[3].name) == "vectorized",
              "fourth registered kernel is vectorized");
          check(kernels[3].launch == gemm::launch_vectorized_tiled,
              "vectorized descriptor uses launch_vectorized_tiled");
          check(kernels[3].author_kernel,
              "vectorized is marked as an author kernel");
          check(std::string_view(kernels[4].name) == "async-16b",
              "fifth registered kernel is async-16b");
          check(kernels[4].launch == gemm::launch_double_buffer,
              "async-16b descriptor uses launch_double_buffer");
          check(kernels[4].author_kernel,
              "async-16b is marked as an author kernel");
          check(std::string_view(kernels[5].name) == "cublas-fp32",
              "sixth registered kernel is cublas-fp32");
          check(kernels[5].launch == gemm::launch_cublas_fp32,
              "cublas descriptor uses launch_cublas_fp32");
          check(!kernels[5].author_kernel,
              "cublas is marked as a vendor baseline");
    }

    const gemm::KernelDescriptor* naive_first = gemm::find_kernel("naive");
    const gemm::KernelDescriptor* naive_second = gemm::find_kernel("naive");
    check(naive_first != nullptr, "find_kernel locates naive");
    check(naive_first == naive_second,
        "find_kernel returns a stable naive descriptor pointer");
    const gemm::KernelDescriptor* shared_first = gemm::find_kernel("shared");
    const gemm::KernelDescriptor* shared_second = gemm::find_kernel("shared");
    check(shared_first != nullptr, "find_kernel locates shared");
    check(shared_first == shared_second,
        "find_kernel returns a stable shared descriptor pointer");
    const gemm::KernelDescriptor* register_first = gemm::find_kernel("register");
    const gemm::KernelDescriptor* register_second = gemm::find_kernel("register");
    check(register_first != nullptr, "find_kernel locates register");
    check(register_first == register_second,
        "find_kernel returns a stable register descriptor pointer");
    const gemm::KernelDescriptor* vectorized_first = gemm::find_kernel("vectorized");
    const gemm::KernelDescriptor* vectorized_second = gemm::find_kernel("vectorized");
    check(vectorized_first != nullptr, "find_kernel locates vectorized");
    check(vectorized_first == vectorized_second,
        "find_kernel returns a stable vectorized descriptor pointer");
    const gemm::KernelDescriptor* async_first = gemm::find_kernel("async-16b");
    const gemm::KernelDescriptor* async_second = gemm::find_kernel("async-16b");
    check(async_first != nullptr, "find_kernel locates async-16b");
    check(async_first == async_second,
        "find_kernel returns a stable async-16b descriptor pointer");
    const gemm::KernelDescriptor* cublas_first = gemm::find_kernel("cublas-fp32");
    const gemm::KernelDescriptor* cublas_second = gemm::find_kernel("cublas-fp32");
    check(cublas_first != nullptr, "find_kernel locates cublas-fp32");
    check(cublas_first == cublas_second,
        "find_kernel returns a stable cublas descriptor pointer");
    check(gemm::find_kernel("Naive") == nullptr,
        "kernel lookup is case-sensitive");
    check(gemm::find_kernel("Shared") == nullptr,
        "shared kernel lookup is case-sensitive");
    check(gemm::find_kernel("Register") == nullptr,
        "register kernel lookup is case-sensitive");
    check(gemm::find_kernel("Vectorized") == nullptr,
        "vectorized kernel lookup is case-sensitive");
    check(gemm::find_kernel("ASYNC-16B") == nullptr,
        "async-16b kernel lookup is case-sensitive");
    check(gemm::find_kernel("CUBLAS-FP32") == nullptr,
        "cublas kernel lookup is case-sensitive");
    check(gemm::find_kernel("missing") == nullptr,
        "unknown kernel returns nullptr");
}

void test_scoped_binding_restores_previous_pointer() {
    FakeKernelState previous_state;
    FakeKernelState scoped_state;

    {
        ScopedBinding previous_binding(active_fake_kernel_state, &previous_state);
        {
            ScopedBinding scoped_binding(active_fake_kernel_state, &scoped_state);
            check(active_fake_kernel_state == &scoped_state,
                  "scoped binding installs replacement pointer");
        }

        check(active_fake_kernel_state == &previous_state,
              "scoped binding restores previous pointer");
    }

    check(active_fake_kernel_state == nullptr,
          "outer scoped binding restores null pointer");
}

void test_gpu_free_cli_paths() {
    std::ostringstream help_output;
    check(gemm::run(parse({"gemm_runner", "--help"}), help_output) == 0,
                    "help succeeds");
    check(help_output.str().find("--warmup <count>       Default: 5") !=
          std::string::npos,
                    "help documents warmup default");
    check(help_output.str().find("Validate mode reports latency_ms=0 and gflops=0") !=
          std::string::npos,
                    "help documents validate timing fields");

    std::ostringstream list_output;
    check(gemm::run(parse({"gemm_runner", "--list"}), list_output) == 0,
                    "kernel list succeeds");
    check(list_output.str() ==
              "naive\nshared\nregister\nvectorized\nasync-16b\ncublas-fp32\n",
          "kernel list contains author kernels followed by cublas-fp32");

    std::ostringstream unknown_output;
    check_throws(
                [&] {
                        gemm::run(parse({"gemm_runner", "--kernel", "missing", "--m", "17",
                                                         "--n", "19", "--k", "23", "--mode", "validate"}),
                                            unknown_output);
                },
                "unknown kernel: missing", "unknown kernel before GPU work");
}

void test_validate_mode_accepts_test_local_kernel_and_reports_pass() {
    FakeKernelState state;
    state.path = "test_reference";
    ScopedBinding state_binding(active_fake_kernel_state, &state);

    const gemm::KernelDescriptor kernel{"fake_reference", fake_reference_launcher,
                                        false};
    const gemm::RunnerOptions options =
        make_run_options(gemm::RunMode::validate, 2, 4, 2, 3, 4);

    std::ostringstream output;
    const int result = runner_internal::run_with_kernel(options, kernel, output);
    check(result == 0, "validate mode returns success for matching fake kernel");
    check(state.launches == 3, "validate mode runs warmups plus one validation launch");

    const std::string text = output.str();
    check(text.find("kernel=fake_reference path=test_reference shape=2x3x4 status=PASS") !=
              std::string::npos,
          "validate mode reports kernel, path, shape, and PASS");
    check(text.find("latency_ms=0") != std::string::npos,
          "validate mode reports zero latency placeholder");
    check(text.find("gflops=0") != std::string::npos,
          "validate mode reports zero gflops placeholder");
    check(text.find("reference=cpu") != std::string::npos,
          "validate mode reports the CPU reference source");
}

void test_reference_selection_uses_exact_work_boundary() {
    const int threshold = static_cast<int>(gemm::kMaxCpuReferenceWork);
    check(runner_internal::uses_cpu_reference(gemm::Problem{1, 1, threshold - 1}),
          "work immediately below threshold uses CPU reference");
    check(!runner_internal::uses_cpu_reference(gemm::Problem{1, 1, threshold}),
          "work at threshold uses large reference");

    const runner_internal::ValidationTolerances large_tolerances =
      runner_internal::validation_tolerances(false);
    check(large_tolerances.atol == 1.0e-3,
        "large reference uses fixed 1e-3 absolute tolerance");
    check(large_tolerances.rtol == 2.0e-3,
        "large reference retains fixed 2e-3 relative tolerance");
}

void test_benchmark_mode_runs_validation_then_warmups_and_iterations() {
    FakeKernelState state;
    state.path = "benchmark_path";
    ScopedBinding state_binding(active_fake_kernel_state, &state);

    const gemm::KernelDescriptor kernel{"fake_reference", fake_reference_launcher,
                                        false};
    const gemm::RunnerOptions options =
        make_run_options(gemm::RunMode::benchmark, 2, 4, 3, 2, 5);

    std::ostringstream output;
    const int result = runner_internal::run_with_kernel(options, kernel, output);
    check(result == 0, "benchmark mode returns success for matching fake kernel");
    check(state.launches == 9,
          "benchmark mode performs validation launch, warmups, and timed iterations");

    const std::string text = output.str();
    check(text.find("kernel=fake_reference path=benchmark_path shape=3x2x5 status=PASS") !=
              std::string::npos,
          "benchmark mode reports PASS with selected path");
    check(text.find("latency_ms=") != std::string::npos,
          "benchmark mode reports latency");
    check(text.find("gflops=") != std::string::npos,
          "benchmark mode reports gflops");
}

void test_large_problem_reference_error_happens_before_allocation() {
    FakeKernelState state;
    ScopedBinding state_binding(active_fake_kernel_state, &state);

    AllocationTracker tracker;
    ScopedBinding tracker_binding(active_allocation_tracker, &tracker);

    const gemm::KernelDescriptor kernel{"fake_reference", fake_reference_launcher,
                                        false};
    const gemm::RunnerOptions options =
        make_run_options(gemm::RunMode::benchmark, 1, 2, 10000, 10000, 2);

    const runner_internal::CudaApi api{
        counting_malloc,
        counting_free,
        counting_memcpy_h2d,
        counting_memcpy_d2h,
        counting_peek_at_last_error,
        counting_device_synchronize,
        counting_event_create,
        counting_event_destroy,
        counting_event_record,
        counting_event_synchronize,
        counting_event_elapsed_time,
    };

    std::ostringstream output;
    check_throws(
        [&] { runner_internal::run_with_kernel(options, kernel, output, api); },
        "large-problem reference is not available yet",
        "benchmark propagates large reference availability error");
    check(tracker.malloc_calls == 0,
          "large-reference failure occurs before any GPU allocation");
    check(state.launches == 0,
          "large-reference failure occurs before any kernel launch");
}

void test_supplied_large_reference_handles_threshold_problem() {
    large_reference_evaluations = 0;
    const gemm::KernelDescriptor kernel{"fake_zero", fake_zero_launcher, false};
    const gemm::RunnerOptions options =
        make_run_options(gemm::RunMode::validate, 0, 1, 1000, 1000, 100);
    const runner_internal::CudaApi api{
        host_malloc,
        host_free,
        host_memcpy,
        host_memcpy,
        counting_peek_at_last_error,
        counting_device_synchronize,
        counting_event_create,
        counting_event_destroy,
        counting_event_record,
        counting_event_synchronize,
        counting_event_elapsed_time,
    };

    std::ostringstream output;
    const int result = runner_internal::run_with_kernel(
        options, kernel, output, api, fake_large_reference);
    check(result == 0, "supplied large reference validates threshold problem");
    check(large_reference_evaluations == 1,
          "large reference is generated exactly once before validation");
    check(output.str().find("reference=cublas-pedantic-fp32") != std::string::npos,
          "large-reference source is visible in output");
}

void test_large_reference_rejects_globally_biased_output() {
    scripted_d2h_evaluations = 0;
    const gemm::KernelDescriptor kernel{"fake_biased", fake_biased_launcher, false};
    const gemm::RunnerOptions options =
        make_run_options(gemm::RunMode::validate, 0, 1, 500, 500, 400);
    const runner_internal::CudaApi api{
        counting_malloc,
        counting_free,
        counting_memcpy_h2d,
        scripted_memcpy_d2h,
        counting_peek_at_last_error,
        counting_device_synchronize,
        counting_event_create,
        counting_event_destroy,
        counting_event_record,
        counting_event_synchronize,
        counting_event_elapsed_time,
    };

    std::ostringstream output;
    const int result = runner_internal::run_with_kernel(
        options, kernel, output, api, fake_counting_large_reference);
    check(result != 0,
          "large reference rejects 0.01 global bias above both tolerances");

    const std::string text = output.str();
    check(text.find("status=FAIL") != std::string::npos,
          "large-reference bias failure is visible in output");
    check(text.find("max_abs=0.010000") != std::string::npos,
          "large-reference bias reports the expected max_abs");
        check(text.find("max_rel=0.010000") != std::string::npos,
            "large-reference bias reports max_rel above 2e-3");
    check(text.find("reference=cublas-pedantic-fp32") != std::string::npos,
          "large-reference bias reports the reference source");
}

void test_validation_failure_returns_nonzero_and_reports_fail() {
    FakeKernelState state;
    state.delta = 0.01F;
    state.path = "failing_path";
    ScopedBinding state_binding(active_fake_kernel_state, &state);

    const gemm::KernelDescriptor kernel{"fake_reference", fake_reference_launcher,
                                        false};
    const gemm::RunnerOptions options =
        make_run_options(gemm::RunMode::validate, 1, 3, 2, 2, 2);

    std::ostringstream output;
    const int result = runner_internal::run_with_kernel(options, kernel, output);
    check(result != 0, "validate mode returns nonzero on validation failure");

    const std::string text = output.str();
    check(text.find("status=FAIL") != std::string::npos,
          "validate mode reports FAIL");
    check(text.find("path=failing_path") != std::string::npos,
          "validate mode still reports selected path on failure");
}

void test_validation_uses_fixed_one_milli_tolerances() {
    const gemm::KernelDescriptor kernel{"fake_reference", fake_reference_launcher,
                                        false};
    const gemm::RunnerOptions options =
        make_run_options(gemm::RunMode::validate, 0, 1, 1, 1, 1);

    FakeKernelState pass_state;
    pass_state.delta = 9.0e-4F;
    {
        ScopedBinding pass_binding(active_fake_kernel_state, &pass_state);
        std::ostringstream pass_output;
        check(runner_internal::run_with_kernel(options, kernel, pass_output) == 0,
              "validation accepts error below 1e-3");
    }

    FakeKernelState fail_state;
    fail_state.delta = 1.2e-3F;
    {
        ScopedBinding fail_binding(active_fake_kernel_state, &fail_state);
        std::ostringstream fail_output;
        check(runner_internal::run_with_kernel(options, kernel, fail_output) != 0,
              "validation rejects error above 1e-3");
        check(fail_output.str().find("status=FAIL") != std::string::npos,
              "fixed tolerance failure is visible in output");
    }
}

void test_csv_logging_creates_header_appends_rows_and_uses_env_metadata() {
    large_reference_evaluations = 0;
    scripted_elapsed_ms = 12.0F;

    const std::filesystem::path csv_path = make_temp_csv_path();
    std::filesystem::remove(csv_path);

    const gemm::KernelDescriptor kernel{"fake_zero", fake_zero_launcher, false};
    gemm::RunnerOptions validate_options =
        make_run_options(gemm::RunMode::validate, 0, 1, 1000, 1000, 100);
    validate_options.csv_path = csv_path.string();

    gemm::RunnerOptions benchmark_options = validate_options;
    benchmark_options.mode = gemm::RunMode::benchmark;
    benchmark_options.iterations = 3;

    const runner_internal::CudaApi api =
        make_host_cuda_api(host_memcpy, scripted_event_elapsed_time);

    ScopedEnvironmentVariable timestamp("GEMM_BENCH_TIMESTAMP", "2026-07-11T08:00:00Z");
    ScopedEnvironmentVariable commit("GEMM_GIT_COMMIT", "deadbeefcafebabe");
    ScopedEnvironmentVariable gpu("GEMM_GPU", "NVIDIA_A100-SXM4-80GB");
    ScopedEnvironmentVariable cuda("GEMM_CUDA", "CUDA_12.4_runtime");
    ScopedEnvironmentVariable nvcc("GEMM_NVCC", "Cuda compilation tools, release 12.4, V12.4.131");

    {
        std::ostringstream output;
        const int result = runner_internal::run_with_kernel(
            validate_options, kernel, output, api, fake_large_reference);
        check(result == 0, "validate CSV invocation succeeds");
    }

    {
        std::ostringstream output;
        const int result = runner_internal::run_with_kernel(
            benchmark_options, kernel, output, api, fake_large_reference);
        check(result == 0, "benchmark CSV invocation succeeds");
    }

    const std::vector<std::string> lines = read_lines(csv_path);
    check(lines.size() == 3, "CSV file contains one header plus two rows");
    if (lines.size() == 3) {
        check(lines[0] ==
                  "timestamp,git_commit,gpu,cuda,nvcc,kernel,path,m,n,k,warmup,iterations,latency_ms,gflops,passed,max_abs,max_rel,reference",
              "CSV header matches the required schema exactly");

        const std::vector<std::string> validate_row = split_csv_row(lines[1]);
        const std::vector<std::string> benchmark_row = split_csv_row(lines[2]);
        check(validate_row.size() == 18, "validate row stores 18 CSV fields");
        check(benchmark_row.size() == 18, "benchmark row stores 18 CSV fields");
        if (validate_row.size() == 18) {
            check(validate_row[0] == "2026-07-11T08:00:00Z", "CSV captures timestamp metadata");
            check(validate_row[1] == "deadbeefcafebabe", "CSV captures git commit metadata");
            check(validate_row[2] == "NVIDIA_A100-SXM4-80GB", "CSV captures GPU metadata");
            check(validate_row[3] == "CUDA_12.4_runtime", "CSV captures CUDA metadata");
            check(validate_row[4] == "Cuda compilation tools, release 12.4, V12.4.131",
                  "CSV captures NVCC metadata");
            check(validate_row[5] == "fake_zero", "CSV stores kernel name");
            check(validate_row[6] == "fake_zero", "CSV stores selected path");
            check(validate_row[7] == "1000" && validate_row[8] == "1000" &&
                      validate_row[9] == "100",
                  "CSV stores benchmark dimensions");
            check(validate_row[10] == "0" && validate_row[11] == "1",
                  "CSV stores warmup and iteration counts");
            check(validate_row[12] == "0.000000" && validate_row[13] == "0.000000",
                  "validate CSV rows retain zero latency and gflops");
            check(validate_row[14] == "true", "validate CSV row records pass=true");
            check(validate_row[17] == "cublas-pedantic-fp32",
                  "validate CSV row records the reference source");
        }

        if (benchmark_row.size() == 18) {
            check(benchmark_row[12] == "4.000000",
                  "benchmark CSV row stores average latency in milliseconds");
            check(std::stod(benchmark_row[13]) > 0.0,
                  "benchmark CSV row stores positive gflops");
        }
    }

    std::filesystem::remove(csv_path);
}

}  // namespace

int main() {
    test_error_macros_throw_readable_single_evaluation_errors();
    test_parser_defaults_and_full_command();
    test_parser_rejects_bad_tokens();
    test_option_validation();
    test_checked_multiply();
    test_input_generation_is_deterministic_and_bounded();
    test_registry_contains_author_kernels_then_cublas_baseline();
    test_scoped_binding_restores_previous_pointer();
    test_gpu_free_cli_paths();
    test_validate_mode_accepts_test_local_kernel_and_reports_pass();
    test_reference_selection_uses_exact_work_boundary();
    test_benchmark_mode_runs_validation_then_warmups_and_iterations();
    test_large_problem_reference_error_happens_before_allocation();
    test_supplied_large_reference_handles_threshold_problem();
    test_large_reference_rejects_globally_biased_output();
    test_validation_failure_returns_nonzero_and_reports_fail();
    test_validation_uses_fixed_one_milli_tolerances();
    test_csv_logging_creates_header_appends_rows_and_uses_env_metadata();

    if (failures != 0) {
        std::cerr << failures << " runner test assertion(s) failed\n";
        return 1;
    }

    std::cout << "All gemm_runner tests passed\n";
    return 0;
}
