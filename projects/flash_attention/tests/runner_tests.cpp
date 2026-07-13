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
    check(defaults.input_pattern == flash_attention::InputPattern::random,
          "default input pattern is random");

    const flash_attention::RunnerOptions options = parse(
        {"flash_attention_runner", "--kernel", "naive", "--n", "37", "--d",
         "24", "--causal", "1", "--input-pattern", "negative-scores", "--mode",
         "benchmark", "--warmup", "0", "--iterations", "7", "--seed", "42"});
    check(options.kernel == "naive", "parser stores kernel");
    check(options.problem.n == 37 && options.problem.d == 24,
          "parser stores dimensions");
    check(options.problem.causal, "parser stores causal mode");
    check(options.input_pattern ==
              flash_attention::InputPattern::negative_scores,
          "parser stores input pattern");
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
        [] { parse({"flash_attention_runner", "--input-pattern", "unknown"}); },
        "invalid input pattern: unknown (expected random, zero-qk, or "
        "negative-scores)",
        "invalid input pattern");
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

void test_special_input_pattern_semantics() {
    const flash_attention::Problem problem{3, 2, false};
    std::vector<float> q(6, 7.0F);
    std::vector<float> k(6, 9.0F);

    flash_attention::apply_input_pattern(flash_attention::InputPattern::zero_qk,
                                         problem, q, k);
    for (float value : q) {
        check(value == 0.0F, "zero-qk clears Q");
    }
    for (float value : k) {
        check(value == 0.0F, "zero-qk clears K");
    }

    flash_attention::apply_input_pattern(
        flash_attention::InputPattern::negative_scores, problem, q, k);
    for (float value : q) {
        check(value == 1.0F, "negative-scores fills Q with positive ones");
    }
    for (float value : k) {
        check(value < 0.0F, "negative-scores fills K with negative values");
    }
    for (int query = 0; query < problem.n; ++query) {
        for (int key = 0; key < problem.n; ++key) {
            float dot = 0.0F;
            for (int feature = 0; feature < problem.d; ++feature) {
                dot +=
                    q[static_cast<std::size_t>(query) * problem.d + feature] *
                    k[static_cast<std::size_t>(key) * problem.d + feature];
            }
            check(dot < 0.0F,
                  "negative-scores produces a strictly negative score matrix");
        }
    }
}

void test_registry_and_workspace() {
    const std::vector<flash_attention::KernelDescriptor> kernels =
        flash_attention::registered_kernels();
    check(kernels.size() == 4,
          "registry contains naive, tiled, tiled-parallel, and tiled-async "
          "kernels");
    if (kernels.size() == 4) {
        const flash_attention::KernelDescriptor &naive = kernels[0];
        check(std::string_view(naive.name) == "naive",
              "registered kernel is naive");
        check(naive.launch == flash_attention::launch_naive_materialized,
              "descriptor uses naive launcher");
        check(naive.author_kernel, "naive is an author kernel");
        check(naive.workspace_bytes({37, 24, false}) ==
                  static_cast<std::size_t>(37 * 37) * sizeof(float),
              "naive workspace is N*N floats");

        const flash_attention::KernelDescriptor &tiled = kernels[1];
        check(std::string_view(tiled.name) == "tiled",
              "second registered kernel is tiled");
        check(tiled.launch == flash_attention::launch_tiled_online,
              "descriptor uses tiled launcher");
        check(tiled.author_kernel, "tiled is an author kernel");
        check(tiled.workspace_bytes({37, 24, false}) == 0,
              "tiled does not materialize N*N workspace");

        const flash_attention::KernelDescriptor &parallel = kernels[2];
        check(std::string_view(parallel.name) == "tiled-parallel",
              "third registered kernel is tiled-parallel");
        check(parallel.launch == flash_attention::launch_tiled_parallel,
              "descriptor uses tiled-parallel launcher");
        check(parallel.author_kernel, "tiled-parallel is an author kernel");
        check(parallel.workspace_bytes({37, 24, false}) == 0,
              "tiled-parallel does not materialize N*N workspace");

        const flash_attention::KernelDescriptor &async = kernels[3];
        check(std::string_view(async.name) == "tiled-async",
              "fourth registered kernel is tiled-async");
        check(async.launch == flash_attention::launch_tiled_async,
              "descriptor uses tiled-async launcher");
        check(async.author_kernel, "tiled-async is an author kernel");
        check(async.workspace_bytes({37, 24, false}) == 0,
              "tiled-async does not materialize N*N workspace");
    }
    check(flash_attention::find_kernel("missing") == nullptr,
          "unknown kernel is not found");
}

} // namespace

int main() {
    test_parser_defaults_and_full_command();
    test_parser_and_option_errors();
    test_checked_multiply_and_input_generation();
    test_special_input_pattern_semantics();
    test_registry_and_workspace();

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }

    std::cout << "All flash_attention_runner tests passed\n";
    return 0;
}
