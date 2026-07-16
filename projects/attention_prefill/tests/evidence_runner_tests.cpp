#include "attention_prefill/evidence_runner.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string &message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Function>
void check_invalid(Function &&function, const std::string &message) {
    try {
        function();
        check(false, message + " did not throw");
    } catch (const std::invalid_argument &) {
    } catch (...) {
        check(false, message + " threw the wrong exception");
    }
}

attention_prefill::evidence::Options
parse(std::initializer_list<const char *> arguments) {
    return attention_prefill::evidence::parse_arguments(
        static_cast<int>(arguments.size()), arguments.begin());
}

void test_defaults_and_full_parse() {
    const auto defaults = parse({"runner"});
    check(defaults.implementation ==
              attention_prefill::evidence::Implementation::br4,
          "default implementation is br4");
    check(defaults.mode == attention_prefill::evidence::RunMode::validate,
          "default mode is validate");
    check(defaults.warmup == 5 && defaults.iterations == 20,
          "default timing controls");
    check(defaults.seed == 1234U, "default seed");
    check(!defaults.metadata_only, "metadata-only is disabled by default");

    const auto metadata_only = parse({"runner", "--metadata-only"});
    check(metadata_only.metadata_only, "parse metadata-only mode");

    const auto options =
        parse({"runner", "--implementation", "br1", "--n", "1024", "--d", "128",
               "--causal", "1", "--mode", "benchmark", "--warmup", "10",
               "--iterations", "50", "--seed", "42"});
    check(options.implementation ==
              attention_prefill::evidence::Implementation::br1,
          "parse br1 implementation");
    check(options.problem.n == 1024 && options.problem.d == 128 &&
              options.problem.causal,
          "parse problem");
    check(options.mode == attention_prefill::evidence::RunMode::benchmark,
          "parse benchmark mode");
    check(options.warmup == 10 && options.iterations == 50,
          "parse timing controls");
    check(options.seed == 42U, "parse seed");
}

void test_metadata_fields_are_safe_key_value_tokens() {
    const attention_prefill::evidence::RuntimeMetadata metadata{
        2, "GPU-01234567-89ab-cdef-0123-456789abcdef", "NVIDIA_A100-SXM4-80GB",
        "8.0", 12080};
    const std::string fields =
        attention_prefill::evidence::format_metadata_fields(metadata);

    check(fields.find("source_sha256=") == 0,
          "metadata starts with embedded source fingerprint");
    check(fields.find(" build_contract=release-sm80-") != std::string::npos,
          "metadata contains the hashed build contract");
    check(fields.find(" build_contract_payload_sha256=") != std::string::npos,
          "metadata contains the build contract payload hash");
    check(fields.find(" device_index=2 ") != std::string::npos,
          "metadata contains CUDA device index");
    check(fields.find(" gpu_uuid=GPU-01234567-89ab-cdef-0123-456789abcdef ") !=
              std::string::npos,
          "metadata contains stable CUDA UUID");
    check(fields.find(" gpu_name=NVIDIA_A100-SXM4-80GB ") != std::string::npos,
          "metadata contains safe GPU name token");
    check(fields.find(" sm=8.0 driver=12080") != std::string::npos,
          "metadata contains CUDA SM and driver");
    check(fields.find_first_of("\n\r\t") == std::string::npos,
          "metadata contains no unsafe whitespace");
}

void test_invalid_arguments() {
    check_invalid([] { parse({"runner", "--implementation", "async"}); },
                  "invalid implementation");
    check_invalid([] { parse({"runner", "--mode", "profile"}); },
                  "invalid mode");
    check_invalid([] { parse({"runner", "--causal", "2"}); }, "invalid causal");
    check_invalid([] { parse({"runner", "--warmup", "-1"}); },
                  "negative warmup");
    check_invalid([] { parse({"runner", "--iterations", "0"}); },
                  "non-positive iterations");
}

void test_theory_ledger() {
    const auto br1 = attention_prefill::evidence::theory_ledger(
        attention_prefill::evidence::Implementation::br1, {128, 64, false});
    const auto br4 = attention_prefill::evidence::theory_ledger(
        attention_prefill::evidence::Implementation::br4, {128, 64, false});
    const auto m2 = attention_prefill::evidence::theory_ledger(
        attention_prefill::evidence::Implementation::m2, {128, 64, false});
    check(br1.cta_count == 128 && br4.cta_count == 32 && m2.cta_count == 32,
          "CTA count");
    check(br1.requested_kv_elements == 2ULL * 128 * 128 * 64,
          "Br1 requested K/V elements");
    check(br4.requested_kv_elements == 2ULL * 32 * 128 * 64,
          "Br4 requested K/V elements");
}

} // namespace

int main() {
    test_defaults_and_full_parse();
    test_invalid_arguments();
    test_metadata_fields_are_safe_key_value_tokens();
    test_theory_ledger();
    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All attention prefill evidence runner tests passed\n";
    return 0;
}