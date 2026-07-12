#include "flash_attention/reference.hpp"
#include "flash_attention/validation.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check_close(float actual, float expected, float tolerance,
                 const std::string& message) {
    check(std::fabs(actual - expected) <= tolerance, message);
}

template <typename Function>
void check_invalid_argument(Function&& function, const std::string& message) {
    try {
        function();
        check(false, message + " did not throw std::invalid_argument");
    } catch (const std::invalid_argument&) {
    } catch (...) {
        check(false, message + " threw the wrong exception type");
    }
}

void test_reference_hand_result() {
    const float q[] = {1.0F, 2.0F};
    const float k[] = {1.0F, 3.0F};
    const float v[] = {10.0F, 20.0F};
    float output[2] = {};

    flash_attention::reference_cpu(
        q, k, v, output, flash_attention::Problem{2, 1, false});

    check_close(output[0], 18.807971F, 1.0e-5F,
                "non-causal query 0 hand result");
    check_close(output[1], 19.820137F, 1.0e-5F,
                "non-causal query 1 hand result");
}

void test_reference_causal_mask() {
    const float q[] = {1.0F, 2.0F};
    const float k[] = {1.0F, 3.0F};
    const float v[] = {10.0F, 20.0F};
    float output[2] = {};

    flash_attention::reference_cpu(
        q, k, v, output, flash_attention::Problem{2, 1, true});

    check_close(output[0], 10.0F, 1.0e-6F,
                "causal query 0 can only attend to itself");
    check_close(output[1], 19.820137F, 1.0e-5F,
                "causal query 1 sees both keys");
}

void test_reference_contract() {
    float value = 1.0F;
    const flash_attention::Problem valid{1, 1, false};
    check_invalid_argument(
        [&] { flash_attention::reference_cpu(nullptr, &value, &value, &value,
                                             valid); },
        "reference null Q");
    check_invalid_argument(
        [&] { flash_attention::reference_cpu(&value, &value, &value, &value,
                                             {0, 1, false}); },
        "reference non-positive N");
    check_invalid_argument(
        [&] { flash_attention::reference_cpu(&value, &value, &value, &value,
                                             {1, 0, false}); },
        "reference non-positive D");
}

void test_validation_metrics() {
    const float expected[] = {1.0F, -2.0F, 4.0F};
    const float actual[] = {1.25F, -3.0F, 4.5F};
    const flash_attention::ErrorMetrics metrics =
        flash_attention::compare(expected, actual, 3);

    check(metrics.max_abs == 1.0, "compare max_abs");
    check(metrics.max_rel == 0.5, "compare max_rel");
    check(metrics.worst_index == 1, "compare worst index");
    check(metrics.finite, "compare finite values");
    check(!flash_attention::passes(metrics, 0.1, 0.1),
          "passes rejects excessive error");
}

}  // namespace

int main() {
    test_reference_hand_result();
    test_reference_causal_mask();
    test_reference_contract();
    test_validation_metrics();

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }

    std::cout << "All flash_attention_common tests passed\n";
    return 0;
}