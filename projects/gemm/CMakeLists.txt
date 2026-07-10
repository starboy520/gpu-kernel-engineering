find_package(CUDAToolkit REQUIRED)

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

add_library(gemm_runner_support STATIC
    runner/runner.cu
    runner/registry.cu
)

target_include_directories(gemm_runner_support
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gemm_runner_support
    PUBLIC
        gemm_common
        CUDA::cudart
        CUDA::cublas
)

# GCC diagnoses NVCC-generated line markers under -Wpedantic, so CUDA host
# compilation uses the warning set that remains clean under warnings-as-errors.
target_compile_options(gemm_runner_support
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

add_executable(gemm_runner_tests
    tests/runner_tests.cpp
)

target_link_libraries(gemm_runner_tests
    PRIVATE
        gemm_runner_support
)

target_compile_options(gemm_runner_tests
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

add_test(NAME gemm_runner_tests COMMAND gemm_runner_tests)

add_executable(gemm_runner
    runner/main.cu
)

target_link_libraries(gemm_runner
    PRIVATE
        gemm_runner_support
)

target_compile_options(gemm_runner
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
