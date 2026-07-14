#pragma once

#include <charconv>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace gpu_kernel {

inline std::size_t checked_multiply(std::size_t left, std::size_t right,
                                    std::string_view description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error("size overflow while computing " +
                                  std::string(description));
    }
    return left * right;
}

inline std::string require_value(int argc, const char *const argv[], int &index,
                                 const std::string &option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + option);
    }
    return argv[++index];
}

template <typename Integer>
Integer parse_integer(const std::string &text, const std::string &option) {
    Integer value{};
    const char *begin = text.data();
    const char *end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec == std::errc::result_out_of_range) {
        throw std::invalid_argument("integer overflow for " + option + ": " +
                                    text);
    }
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument("invalid integer for " + option + ": " +
                                    text);
    }
    return value;
}

} // namespace gpu_kernel
