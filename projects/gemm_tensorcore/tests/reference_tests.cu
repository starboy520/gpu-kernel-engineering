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

void test_direct_col_major_reference() {
    const std::vector<__half> a = {__float2half_rn(1.0F), __float2half_rn(2.0F),
                                   __float2half_rn(3.0F),
                                   __float2half_rn(4.0F)};
    // Logical B = [[5, 6], [7, 8]], physically stored column-major.
    const std::vector<__half> b = {__float2half_rn(5.0F), __float2half_rn(7.0F),
                                   __float2half_rn(6.0F),
                                   __float2half_rn(8.0F)};
    std::vector<float> c(4);

    gemm_tensorcore::reference_wmma_direct(a.data(), 2, b.data(), 2, c.data(),
                                           2, {2, 2, 2});
    const float expected[] = {19.0F, 22.0F, 43.0F, 50.0F};
    for (std::size_t index = 0; index < c.size(); ++index) {
        check(c[index] == expected[index],
              "direct reference reads col-major B at " + std::to_string(index));
    }
}

void test_direct_padding_zero_semantics() {
    const gemm_tensorcore::Problem logical{17, 19, 23};
    const gemm_tensorcore::Problem padded =
        gemm_tensorcore::pad_problem(logical);
    check(padded.m == 32 && padded.n == 32 && padded.k == 32,
          "17x19x23 pads to 32x32x32");

    std::vector<__half> a(static_cast<std::size_t>(padded.m) * padded.k,
                          __float2half_rn(0.0F));
    std::vector<__half> b(static_cast<std::size_t>(padded.k) * padded.n,
                          __float2half_rn(0.0F));
    std::vector<float> c(static_cast<std::size_t>(padded.m) * padded.n);
    for (int row = 0; row < logical.m; ++row) {
        for (int inner = 0; inner < logical.k; ++inner) {
            a[static_cast<std::size_t>(row) * padded.k + inner] =
                __float2half_rn(1.0F);
        }
    }
    for (int column = 0; column < logical.n; ++column) {
        for (int inner = 0; inner < logical.k; ++inner) {
            b[static_cast<std::size_t>(inner) +
              static_cast<std::size_t>(column) * padded.k] =
                __float2half_rn(1.0F);
        }
    }

    gemm_tensorcore::reference_wmma_direct(
        a.data(), padded.k, b.data(), padded.k, c.data(), padded.n, padded);
    for (int row = 0; row < padded.m; ++row) {
        for (int column = 0; column < padded.n; ++column) {
            const float expected = row < logical.m && column < logical.n
                                       ? static_cast<float>(logical.k)
                                       : 0.0F;
            check(c[static_cast<std::size_t>(row) * padded.n + column] ==
                      expected,
                  "padding contributes zero at (" + std::to_string(row) + "," +
                      std::to_string(column) + ")");
        }
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
    check_invalid_argument(
        [] { (void)gemm_tensorcore::pad_problem({0, 16, 16}); },
        "non-positive padded M");
    check_invalid_argument(
        [&] {
            gemm_tensorcore::reference_wmma_direct(
                &half_value, 1, &half_value, 1, &float_value, 1, {1, 2, 1});
        },
        "invalid Direct ldc");
}

} // namespace

int main() {
    test_ones();
    test_identity_preserves_quantized_b();
    test_direct_col_major_reference();
    test_direct_padding_zero_semantics();
    test_contract();
    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All gemm_tensorcore reference tests passed\n";
    return 0;
}
