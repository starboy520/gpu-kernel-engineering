#include "attention_prefill/evidence_runner.hpp"

#include <exception>
#include <iostream>

int main(int argc, const char *const argv[]) {
    try {
        const auto options =
            attention_prefill::evidence::parse_arguments(argc, argv);
        attention_prefill::evidence::validate_options(options);
        return attention_prefill::evidence::run(options, std::cout);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}