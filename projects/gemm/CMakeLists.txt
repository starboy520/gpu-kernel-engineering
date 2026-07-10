add_library(gemm_common STATIC
    common/reference.cpp
    common/validation.cpp
)

target_include_directories(gemm_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(gemm_common
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_executable(gemm_common_tests
    tests/common_tests.cpp
)

target_link_libraries(gemm_common_tests
    PRIVATE
        gemm_common
)

target_compile_options(gemm_common_tests
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_test(NAME gemm_common_tests COMMAND gemm_common_tests)
