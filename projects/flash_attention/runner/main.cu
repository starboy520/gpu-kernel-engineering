#include "flash_attention/runner.hpp"

#include <exception>
#include <iostream>

int main(int argc, const char* const argv[]) {
    try {
        const flash_attention::RunnerOptions options =
            flash_attention::parse_arguments(argc, argv);
        return flash_attention::run(options, std::cout);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}