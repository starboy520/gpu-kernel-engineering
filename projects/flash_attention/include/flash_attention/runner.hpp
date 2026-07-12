#pragma once

#include "flash_attention/kernel.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace flash_attention {

enum class RunMode {
    validate,
    benchmark,
};

enum class InputPattern {
    random,
    zero_qk,
    negative_scores,
};

struct RunnerOptions {
    bool help = false;
    bool list = false;
    std::string kernel;
    Problem problem{0, 0, false};
    RunMode mode = RunMode::validate;
    int warmup = 5;
    int iterations = 20;
    std::uint32_t seed = 1234U;
    InputPattern input_pattern = InputPattern::random;
};

RunnerOptions parse_arguments(int argc, const char *const argv[]);
void validate_options(const RunnerOptions &options);
std::size_t checked_multiply(std::size_t left, std::size_t right,
                             std::string_view description);
std::vector<float> generate_input(std::size_t count, std::uint32_t seed);
void apply_input_pattern(InputPattern input_pattern, Problem problem,
                         std::vector<float> &q, std::vector<float> &k);
int run(const RunnerOptions &options, std::ostream &output);

} // namespace flash_attention