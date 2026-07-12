#include "flash_attention/kernel.hpp"
#include "flash_attention/runner.hpp"

#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string &message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Function>
void check_throws(Function &&function, const std::string &expected_message,
                  const std::string &description) {
    try {
        function();
        check(false, description + " did not throw");
    } catch (const std::exception &error) {
        check(error.what() == expected_message,
              description + " message was: " + error.what());
    }
}

flash_attention::RunnerOptions
parse(std::initializer_list<const char *> arguments) {
    const std::vector<const char *> argv(arguments);
    return flash_attention::parse_arguments(static_cast<int>(argv.size()),
                                            argv.data());
}

void test_parser_defaults_and_full_command() {
    const flash_attention::RunnerOptions defaults =
        parse({"flash_attention_runner"});
    check(defaults.mode == flash_attention::RunMode::validate,
          "default mode is validate");
    check(defaults.warmup == 5, "default warmup is 5");
    check(defaults.iterations == 20, "default iterations is 20");
    check(defaults.seed == 1234U, "default seed is 1234");
    check(!defaults.problem.causal, "default attention is non-causal");

    const flash_attention::RunnerOptions options =
        parse({"flash_attention_runner", "--kernel", "naive", "--n", "37",
               "--d", "24", "--causal", "1", "--mode", "benchmark", "--warmup",
               "0", "--iterations", "7", "--seed", "42"});
    check(options.kernel == "naive", "parser stores kernel");
    check(options.problem.n == 37 && options.problem.d == 24,
          "parser stores dimensions");
    check(options.problem.causal, "parser stores causal mode");
    check(options.mode == flash_attention::RunMode::benchmark,
          "parser stores benchmark mode");
    check(options.warmup == 0 && options.iterations == 7,
          "parser stores counts");
    check(options.seed == 42U, "parser stores seed");
}

void test_parser_and_option_errors() {
    check_throws([] { parse({"flash_attention_runner", "--causal", "2"}); },
                 "--causal must be 0 or 1", "invalid causal value");
    check_throws(
        [] {
            flash_attention::validate_options(
                parse({"flash_attention_runner", "--kernel", "naive", "--n",
                       "8", "--d", "129"}));
        },
        "--d must be <= 128 in the first version", "unsupported D");
    check_throws(
        [] {
            flash_attention::validate_options(
                parse({"flash_attention_runner"}));
        },
        "normal run requires --kernel", "missing kernel");
}

void test_checked_multiply_and_input_generation() {
    check(flash_attention::checked_multiply(7, 9, "tensor elements") == 63,
          "checked_multiply computes product");
    check_throws(
        [] {
            flash_attention::checked_multiply(
                std::numeric_limits<std::size_t>::max(), 2, "tensor bytes");
        },
        "size overflow while computing tensor bytes",
        "checked multiplication overflow");

    const std::vector<float> first = flash_attention::generate_input(128, 42U);
    const std::vector<float> second = flash_attention::generate_input(128, 42U);
    const std::vector<float> different =
        flash_attention::generate_input(128, 43U);
    check(first == second, "input generation is deterministic");
    check(first != different, "different seeds produce different input");
    for (float value : first) {
        check(value >= -0.5F && value <= 0.5F, "input value is in [-0.5, 0.5]");
    }
}

void test_registry_and_workspace() {
    const std::vector<flash_attention::KernelDescriptor> kernels =
        flash_attention::registered_kernels();
    check(kernels.size() == 1, "registry contains one learning kernel");
    if (kernels.size() == 1) {
        const flash_attention::KernelDescriptor &kernel = kernels[0];
        check(std::string_view(kernel.name) == "naive",
              "registered kernel is naive");
        check(kernel.launch == flash_attention::launch_naive_materialized,
              "descriptor uses naive launcher");
        check(kernel.author_kernel, "naive is an author kernel");
        check(kernel.workspace_bytes({37, 24, false}) ==
                  static_cast<std::size_t>(37 * 37) * sizeof(float),
              "naive workspace is N*N floats");
    }
    check(flash_attention::find_kernel("missing") == nullptr,
          "unknown kernel is not found");
}

} // namespace

int main() {
    test_parser_defaults_and_full_command();
    test_parser_and_option_errors();
    test_checked_multiply_and_input_generation();
    test_registry_and_workspace();

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }

    std::cout << "All flash_attention_runner tests passed\n";
    return 0;
}
