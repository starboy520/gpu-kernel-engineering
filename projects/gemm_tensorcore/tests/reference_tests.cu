#include "gemm_tensorcore/reference.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
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
void check_invalid_argument(Function &&function, const std::string &message) {
    try {
        function();
        check(false, message + " did not throw");
    } catch (const std::invalid_argument &) {
    } catch (...) {
        check(false, message + " threw the wrong exception");
    }
}

void test_ones() {
    std::vector<__half> a(gemm_tensorcore::wmma_m * gemm_tensorcore::wmma_k,
                          __float2half_rn(1.0F));
    std::vector<__half> b(gemm_tensorcore::wmma_k * gemm_tensorcore::wmma_n,
                          __float2half_rn(1.0F));
    std::vector<float> c(gemm_tensorcore::wmma_m * gemm_tensorcore::wmma_n);
    gemm_tensorcore::reference_wmma_single(a.data(), b.data(), c.data());
    for (float value : c) {
        check(value == 16.0F, "ones output equals 16");
    }
}

void test_identity_preserves_quantized_b() {
    std::vector<__half> a(gemm_tensorcore::wmma_m * gemm_tensorcore::wmma_k,
                          __float2half_rn(0.0F));
    std::vector<__half> b(gemm_tensorcore::wmma_k * gemm_tensorcore::wmma_n);
    std::vector<float> c(gemm_tensorcore::wmma_m * gemm_tensorcore::wmma_n);
    for (int index = 0; index < gemm_tensorcore::wmma_m; ++index) {
        a[index * gemm_tensorcore::wmma_k + index] = __float2half_rn(1.0F);
    }
    for (std::size_t index = 0; index < b.size(); ++index) {
        b[index] =
            __float2half_rn((static_cast<int>(index % 31) - 15) * 0.125F);
    }

    gemm_tensorcore::reference_wmma_single(a.data(), b.data(), c.data());
    for (std::size_t index = 0; index < c.size(); ++index) {
        check(c[index] == __half2float(b[index]),
              "identity preserves quantized B at " + std::to_string(index));
    }
}

void test_contract() {
    __half half_value = __float2half_rn(1.0F);
    float float_value = 0.0F;
    check_invalid_argument(
        [&] {
            gemm_tensorcore::reference_wmma_single(nullptr, &half_value,
                                                   &float_value);
        },
        "null A");
    check_invalid_argument(
        [&] {
            gemm_tensorcore::reference_wmma_single(&half_value, nullptr,
                                                   &float_value);
        },
        "null B");
    check_invalid_argument(
        [&] {
            gemm_tensorcore::reference_wmma_single(&half_value, &half_value,
                                                   nullptr);
        },
        "null C");
}

} // namespace

int main() {
    test_ones();
    test_identity_preserves_quantized_b();
    test_contract();
    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All gemm_tensorcore reference tests passed\n";
    return 0;
}
