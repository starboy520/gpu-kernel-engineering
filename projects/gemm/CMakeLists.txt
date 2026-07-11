find_package(CUDAToolkit REQUIRED)

add_library(gemm_kernels STATIC
    kernels/naive.cu
    kernels/shared_tiled.cu
    kernels/register_tiled.cu
)

target_include_directories(gemm_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gemm_kernels
    PUBLIC
        CUDA::cudart
)

target_compile_options(gemm_kernels
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-warn-spills;-Xcompiler=-Wall,-Wextra>
)

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
        gemm_kernels
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
        gemm_kernels
)

target_compile_options(gemm_runner
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

add_test(
    NAME gemm_naive_rectangular_validate
    COMMAND gemm_runner --kernel naive --m 17 --n 19 --k 23 --mode validate
)

set_tests_properties(gemm_naive_rectangular_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=naive path=naive shape=17x19x23 status=PASS"
)

add_test(
    NAME gemm_shared_rectangular_validate
    COMMAND gemm_runner --kernel shared --m 17 --n 19 --k 23 --mode validate
)

set_tests_properties(gemm_shared_rectangular_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=shared path=shared shape=17x19x23 status=PASS"
)

add_test(
    NAME gemm_shared_k_tail_validate
    COMMAND gemm_runner --kernel shared --m 33 --n 65 --k 17 --mode validate
)

set_tests_properties(gemm_shared_k_tail_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=shared path=shared shape=33x65x17 status=PASS"
)

add_test(
    NAME gemm_register_single_tile_validate
    COMMAND gemm_runner --kernel register --m 64 --n 64 --k 16 --mode validate
)

set_tests_properties(gemm_register_single_tile_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=register path=register shape=64x64x16 status=PASS"
)

add_test(
    NAME gemm_register_rectangular_tail_validate
    COMMAND gemm_runner --kernel register --m 65 --n 127 --k 33 --mode validate
)

set_tests_properties(gemm_register_rectangular_tail_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=register path=register shape=65x127x33 status=PASS"
)
