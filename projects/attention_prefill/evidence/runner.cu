#include "attention_prefill/evidence_runner.hpp"

#include "attention_prefill/warp_per_query.hpp"
#include "flash_attention/kernel.hpp"
#include "gpu_kernel/cuda_check.hpp"
#include "gpu_kernel/runner_utils.hpp"
#include "gpu_kernel/validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace attention_prefill::evidence {
namespace {

constexpr double validation_atol = 2.0e-4;
constexpr double validation_rtol = 2.0e-3;

#ifndef ATTENTION_PREFILL_EVIDENCE_SOURCE_SHA256
#error "ATTENTION_PREFILL_EVIDENCE_SOURCE_SHA256 must be defined by CMake"
#endif

#ifndef ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT
#error "ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT must be defined by CMake"
#endif

#ifndef ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT_PAYLOAD_SHA256
#error                                                                         \
    "ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT_PAYLOAD_SHA256 must be defined by CMake"
#endif

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) {
        GPU_CUDA_CHECK(cudaMalloc(&data_, bytes));
    }
    ~DeviceBuffer() {
        if (data_ != nullptr)
            (void)cudaFree(data_);
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    float *data() const { return static_cast<float *>(data_); }

  private:
    void *data_ = nullptr;
};

class Event {
  public:
    Event() { GPU_CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() {
        if (event_ != nullptr)
            (void)cudaEventDestroy(event_);
    }
    cudaEvent_t get() const { return event_; }

  private:
    cudaEvent_t event_ = nullptr;
};

const char *implementation_name(Implementation implementation) {
    switch (implementation) {
    case Implementation::br1:
        return "br1";
    case Implementation::br4:
        return "br4";
    case Implementation::m2:
        return "m2";
    }
    throw std::logic_error("unknown evidence implementation");
}

std::string safe_token(const char *text) {
    std::string token;
    for (const unsigned char character : std::string(text)) {
        const bool safe = (character >= 'a' && character <= 'z') ||
                          (character >= 'A' && character <= 'Z') ||
                          (character >= '0' && character <= '9') ||
                          character == '.' || character == '_' ||
                          character == '-';
        token.push_back(safe ? static_cast<char>(character) : '_');
    }
    return token;
}

void launch(Implementation implementation, const float *q, const float *k,
            const float *v, float *output, Problem problem) {
    if (implementation == Implementation::br1) {
        const flash_attention::Problem baseline_problem{problem.n, problem.d,
                                                        problem.causal};
        (void)flash_attention::launch_tiled_online(q, k, v, output, nullptr,
                                                   baseline_problem, nullptr);
        return;
    }
    if (implementation == Implementation::br4) {
        launch_query_tiled(q, k, v, output, problem, nullptr);
        return;
    }
    launch_warp_per_query(q, k, v, output, problem, nullptr);
}

std::vector<float> make_input(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    std::vector<float> values(count);
    for (float &value : values)
        value = distribution(generator);
    return values;
}

void reference(const std::vector<float> &q, const std::vector<float> &k,
               const std::vector<float> &v, std::vector<float> &output,
               Problem problem) {
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
            scores[static_cast<std::size_t>(key)] = dot * scale;
            row_max = std::max(row_max, dot * scale);
        }
        double denominator = 0.0;
        for (double &score : scores) {
            if (std::isinf(score) && score < 0.0)
                score = 0.0;
            else {
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

} // namespace

Options parse_arguments(int argc, const char *const argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        const auto value = [&] {
            return gpu_kernel::require_value(argc, argv, index, option);
        };
        if (option == "--implementation") {
            const std::string text = value();
            if (text == "br1")
                options.implementation = Implementation::br1;
            else if (text == "br4")
                options.implementation = Implementation::br4;
            else if (text == "m2")
                options.implementation = Implementation::m2;
            else
                throw std::invalid_argument(
                    "--implementation must be br1, br4, or m2");
        } else if (option == "--n")
            options.problem.n = gpu_kernel::parse_integer<int>(value(), option);
        else if (option == "--d")
            options.problem.d = gpu_kernel::parse_integer<int>(value(), option);
        else if (option == "--causal") {
            const int causal = gpu_kernel::parse_integer<int>(value(), option);
            if (causal != 0 && causal != 1)
                throw std::invalid_argument("--causal must be 0 or 1");
            options.problem.causal = causal == 1;
        } else if (option == "--mode") {
            const std::string text = value();
            if (text == "validate")
                options.mode = RunMode::validate;
            else if (text == "benchmark")
                options.mode = RunMode::benchmark;
            else
                throw std::invalid_argument(
                    "--mode must be validate or benchmark");
        } else if (option == "--warmup")
            options.warmup = gpu_kernel::parse_integer<int>(value(), option);
        else if (option == "--iterations")
            options.iterations =
                gpu_kernel::parse_integer<int>(value(), option);
        else if (option == "--seed")
            options.seed =
                gpu_kernel::parse_integer<std::uint32_t>(value(), option);
        else if (option == "--metadata-only")
            options.metadata_only = true;
        else
            throw std::invalid_argument("unknown option: " + option);
    }
    if (options.warmup < 0 || options.iterations <= 0)
        throw std::invalid_argument(
            "warmup must be nonnegative and iterations positive");
    return options;
}

void validate_options(const Options &options) {
    if (options.metadata_only)
        return;
    if (options.problem.n <= 0 || options.problem.d <= 0 ||
        options.problem.d > max_head_dimension)
        throw std::invalid_argument("require N>0 and 1<=D<=128");
}

RuntimeMetadata query_runtime_metadata() {
    RuntimeMetadata metadata{};
    GPU_CUDA_CHECK(cudaGetDevice(&metadata.device_index));
    cudaDeviceProp properties{};
    GPU_CUDA_CHECK(cudaGetDeviceProperties(&properties, metadata.device_index));
    metadata.gpu_name = safe_token(properties.name);

    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int index = 0; index < 16; ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10)
            uuid << '-';
        uuid << std::setw(2)
             << static_cast<unsigned int>(
                    static_cast<unsigned char>(properties.uuid.bytes[index]));
    }
    metadata.gpu_uuid = uuid.str();
    metadata.sm = std::to_string(properties.major) + "." +
                  std::to_string(properties.minor);
    GPU_CUDA_CHECK(cudaDriverGetVersion(&metadata.driver));
    return metadata;
}

std::string format_metadata_fields(const RuntimeMetadata &metadata) {
    std::ostringstream output;
    output << "source_sha256=" ATTENTION_PREFILL_EVIDENCE_SOURCE_SHA256
           << " build_contract=" ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT
           << " build_contract_payload_"
              "sha256=" ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT_PAYLOAD_SHA256
           << " device_index=" << metadata.device_index
           << " gpu_uuid=" << metadata.gpu_uuid
           << " gpu_name=" << metadata.gpu_name << " sm=" << metadata.sm
           << " driver=" << metadata.driver;
    return output.str();
}

TheoryLedger theory_ledger(Implementation implementation, Problem problem) {
    const std::uint64_t n = static_cast<std::uint64_t>(problem.n);
    const std::uint64_t d = static_cast<std::uint64_t>(problem.d);
    const std::uint64_t ctas =
        implementation == Implementation::br1 ? n : (n + 3) / 4;
    return {ctas, 2ULL * ctas * n * d};
}

int run(const Options &options, std::ostream &output) {
    validate_options(options);
    const RuntimeMetadata metadata = query_runtime_metadata();
    if (options.metadata_only) {
        output << format_metadata_fields(metadata) << '\n';
        return 0;
    }
    const std::size_t count = gpu_kernel::checked_multiply(
        static_cast<std::size_t>(options.problem.n),
        static_cast<std::size_t>(options.problem.d), "attention tensor");
    const std::size_t bytes =
        gpu_kernel::checked_multiply(count, sizeof(float), "attention bytes");
    const auto q = make_input(count, options.seed);
    const auto k = make_input(count, options.seed + 1U);
    const auto v = make_input(count, options.seed + 2U);
    std::vector<float> expected(count), actual(count);
    reference(q, k, v, expected, options.problem);
    DeviceBuffer device_q(bytes), device_k(bytes), device_v(bytes),
        device_output(bytes);
    GPU_CUDA_CHECK(
        cudaMemcpy(device_q.data(), q.data(), bytes, cudaMemcpyHostToDevice));
    GPU_CUDA_CHECK(
        cudaMemcpy(device_k.data(), k.data(), bytes, cudaMemcpyHostToDevice));
    GPU_CUDA_CHECK(
        cudaMemcpy(device_v.data(), v.data(), bytes, cudaMemcpyHostToDevice));
    GPU_CUDA_CHECK(cudaMemset(device_output.data(), 0xFF, bytes));
    launch(options.implementation, device_q.data(), device_k.data(),
           device_v.data(), device_output.data(), options.problem);
    GPU_CUDA_CHECK(cudaDeviceSynchronize());
    GPU_CUDA_CHECK(cudaMemcpy(actual.data(), device_output.data(), bytes,
                              cudaMemcpyDeviceToHost));
    const auto metrics =
        gpu_kernel::compare(expected.data(), actual.data(), count);
    const bool passed =
        gpu_kernel::passes(metrics, validation_atol, validation_rtol);
    double latency_ms = 0.0;
    if (passed && options.mode == RunMode::benchmark) {
        for (int i = 0; i < options.warmup; ++i)
            launch(options.implementation, device_q.data(), device_k.data(),
                   device_v.data(), device_output.data(), options.problem);
        GPU_CUDA_CHECK(cudaDeviceSynchronize());
        Event start, stop;
        GPU_CUDA_CHECK(cudaEventRecord(start.get()));
        for (int i = 0; i < options.iterations; ++i)
            launch(options.implementation, device_q.data(), device_k.data(),
                   device_v.data(), device_output.data(), options.problem);
        GPU_CUDA_CHECK(cudaEventRecord(stop.get()));
        GPU_CUDA_CHECK(cudaEventSynchronize(stop.get()));
        float total_ms = 0.0F;
        GPU_CUDA_CHECK(
            cudaEventElapsedTime(&total_ms, start.get(), stop.get()));
        latency_ms = static_cast<double>(total_ms) / options.iterations;
    }
    const auto ledger = theory_ledger(options.implementation, options.problem);
    output << format_metadata_fields(metadata) << ' ' << std::fixed
           << std::setprecision(6)
           << "implementation=" << implementation_name(options.implementation)
           << " path=" << implementation_name(options.implementation)
           << " shape=" << options.problem.n << 'x' << options.problem.d
           << " causal=" << (options.problem.causal ? 1 : 0)
           << " input_pattern=random status=" << (passed ? "PASS" : "FAIL")
           << " max_abs=" << metrics.max_abs << " max_rel=" << metrics.max_rel
           << " latency_ms=" << latency_ms << " cta_count=" << ledger.cta_count
           << " requested_kv_elements=" << ledger.requested_kv_elements
           << " workspace_bytes=0\n";
    return passed ? 0 : 1;
}

} // namespace attention_prefill::evidence