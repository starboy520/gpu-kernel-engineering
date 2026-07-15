#pragma once

#include "attention_prefill/query_tiled.hpp"

#include <cstdint>
#include <iosfwd>
#include <string>

namespace attention_prefill::evidence {

enum class Implementation { br1, br4 };
enum class RunMode { validate, benchmark };

struct Options {
    Implementation implementation = Implementation::br4;
    Problem problem{0, 0, false};
    RunMode mode = RunMode::validate;
    int warmup = 5;
    int iterations = 20;
    std::uint32_t seed = 1234U;
    bool metadata_only = false;
};

struct TheoryLedger {
    std::uint64_t cta_count;
    std::uint64_t requested_kv_elements;
};

struct RuntimeMetadata {
    int device_index;
    std::string gpu_uuid;
    std::string gpu_name;
    std::string sm;
    int driver;
};

Options parse_arguments(int argc, const char *const argv[]);
void validate_options(const Options &options);
TheoryLedger theory_ledger(Implementation implementation, Problem problem);
RuntimeMetadata query_runtime_metadata();
std::string format_metadata_fields(const RuntimeMetadata &metadata);
int run(const Options &options, std::ostream &output);

} // namespace attention_prefill::evidence