#include "attention_prefill/query_tiled.hpp"

#include <iostream>
#include <limits>
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

template <typename Exception, typename Function>
void check_throws(Function &&function, const std::string &message) {
    try {
        function();
        check(false, message + " did not throw");
    } catch (const Exception &) {
    } catch (...) {
        check(false, message + " threw the wrong exception type");
    }
}

void test_problem_contract() {
    check_throws<std::invalid_argument>(
        [] {
            attention_prefill::launch_query_tiled(
                nullptr, nullptr, nullptr, nullptr, {0, 64, false}, nullptr);
        },
        "non-positive N");
    check_throws<std::invalid_argument>(
        [] {
            attention_prefill::launch_query_tiled(
                nullptr, nullptr, nullptr, nullptr, {1, 0, false}, nullptr);
        },
        "non-positive D");
    check_throws<std::invalid_argument>(
        [] {
            attention_prefill::launch_query_tiled(
                nullptr, nullptr, nullptr, nullptr, {1, 129, false}, nullptr);
        },
        "D above static capacity");
    check_throws<std::overflow_error>(
        [] {
            attention_prefill::launch_query_tiled(
                nullptr, nullptr, nullptr, nullptr,
                {std::numeric_limits<int>::max(), 128, false}, nullptr);
        },
        "32-bit device index overflow");
    check_throws<std::overflow_error>(
        [] {
            attention_prefill::launch_query_tiled(
                nullptr, nullptr, nullptr, nullptr,
                {std::numeric_limits<int>::max(), 1, false}, nullptr);
        },
        "K/V tile loop index overflow");
    check_throws<std::invalid_argument>(
        [] {
            attention_prefill::launch_query_tiled(
                nullptr, nullptr, nullptr, nullptr, {1, 1, false}, nullptr);
        },
        "null buffers");
}

} // namespace

int main() {
    test_problem_contract();
    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All attention_prefill contract tests passed\n";
    return 0;
}
