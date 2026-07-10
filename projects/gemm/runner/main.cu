#include "gemm/runner.hpp"

#include <exception>
#include <iostream>

int main(int argc, const char* const argv[]) {
    try {
        const gemm::RunnerOptions options = gemm::parse_arguments(argc, argv);
        return gemm::run(options, std::cout);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
