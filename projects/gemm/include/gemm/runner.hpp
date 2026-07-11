#pragma once

#include "gemm/kernel.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace gemm {

enum class RunMode {
    validate,
    benchmark,
};

struct RunnerOptions {
    bool help = false;
    bool list = false;
    std::string kernel;
    Problem problem{0, 0, 0};
    RunMode mode = RunMode::validate;
    int warmup = 5;
    int iterations = 20;
    std::uint32_t seed = 1234U;
    std::string csv_path;
};

inline constexpr std::size_t kMaxCpuReferenceWork = 100'000'000;

RunnerOptions parse_arguments(int argc, const char* const argv[]);
void validate_options(const RunnerOptions& options);
std::size_t checked_multiply(std::size_t left, std::size_t right,
                             std::string_view description);
std::vector<float> generate_input(std::size_t count, std::uint32_t seed);
int run(const RunnerOptions& options, std::ostream& output);

namespace runner_internal {

bool uses_cpu_reference(Problem problem);
struct ValidationTolerances {
    double atol;
    double rtol;
};
ValidationTolerances validation_tolerances(bool cpu_reference);
using LargeReferenceFn = void (*)(const float*, const float*, float*, Problem,
                                  cudaStream_t);

struct CudaApi {
    cudaError_t (*malloc_fn)(void**, std::size_t);
    cudaError_t (*free_fn)(void*);
    cudaError_t (*memcpy_h2d_fn)(void*, const void*, std::size_t);
    cudaError_t (*memcpy_d2h_fn)(void*, const void*, std::size_t);
    cudaError_t (*peek_at_last_error_fn)();
    cudaError_t (*device_synchronize_fn)();
    cudaError_t (*event_create_fn)(cudaEvent_t*);
    cudaError_t (*event_destroy_fn)(cudaEvent_t);
    cudaError_t (*event_record_fn)(cudaEvent_t, cudaStream_t);
    cudaError_t (*event_synchronize_fn)(cudaEvent_t);
    cudaError_t (*event_elapsed_time_fn)(float*, cudaEvent_t, cudaEvent_t);
};

int run_with_kernel(const RunnerOptions& options, const KernelDescriptor& kernel,
                    std::ostream& output);
int run_with_kernel(const RunnerOptions& options, const KernelDescriptor& kernel,
                    std::ostream& output, const CudaApi& cuda_api);
int run_with_kernel(const RunnerOptions& options, const KernelDescriptor& kernel,
                    std::ostream& output, const CudaApi& cuda_api,
                    LargeReferenceFn large_reference);

}  // namespace runner_internal

}  // namespace gemm
