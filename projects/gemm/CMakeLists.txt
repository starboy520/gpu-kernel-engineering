find_package(CUDAToolkit REQUIRED)

add_library(gemm_kernels STATIC
    kernels/naive.cu
    kernels/shared_tiled.cu
    kernels/register_tiled.cu
    kernels/vectorized.cu
    kernels/double_buffer.cu
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

target_link_libraries(gemm_common
    PUBLIC
        CUDA::cudart
)

target_compile_options(gemm_common
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_library(gemm_vendor_baseline STATIC
    common/reference_cublas.cu
)

target_include_directories(gemm_vendor_baseline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gemm_vendor_baseline
    PUBLIC
        CUDA::cublas
        CUDA::cudart
)

target_compile_options(gemm_vendor_baseline
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xcompiler=-Wall,-Wextra>
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
        gemm_vendor_baseline
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

add_test(
    NAME gemm_vectorized_fast_aligned_validate
    COMMAND gemm_runner --kernel vectorized --m 128 --n 128 --k 128 --mode validate
)

set_tests_properties(gemm_vectorized_fast_aligned_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=vectorized path=fast-float4 shape=128x128x128 status=PASS.*reference=cpu"
)

add_test(
    NAME gemm_vectorized_m_tail_fast_validate
    COMMAND gemm_runner --kernel vectorized --m 130 --n 128 --k 128 --mode validate
)

set_tests_properties(gemm_vectorized_m_tail_fast_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=vectorized path=fast-float4 shape=130x128x128 status=PASS.*reference=cpu"
)

add_test(
    NAME gemm_vectorized_n_nonmultiple_fallback_validate
    COMMAND gemm_runner --kernel vectorized --m 128 --n 130 --k 128 --mode validate
)

set_tests_properties(gemm_vectorized_n_nonmultiple_fallback_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=vectorized path=fallback-register shape=128x130x128 status=PASS.*reference=cpu"
)

add_test(
    NAME gemm_vectorized_k_nonmultiple_fallback_validate
    COMMAND gemm_runner --kernel vectorized --m 128 --n 128 --k 130 --mode validate
)

set_tests_properties(gemm_vectorized_k_nonmultiple_fallback_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=vectorized path=fallback-register shape=128x128x130 status=PASS.*reference=cpu"
)

add_test(
    NAME gemm_async_16b_single_tile_validate
    COMMAND gemm_runner --kernel async-16b --m 64 --n 64 --k 16 --mode validate
)

set_tests_properties(gemm_async_16b_single_tile_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=async-16b path=fast-pipeline-16b shape=64x64x16 status=PASS.*reference=cpu"
)

add_test(
    NAME gemm_async_16b_m_tail_multitile_validate
    COMMAND gemm_runner --kernel async-16b --m 65 --n 128 --k 32 --mode validate
)

set_tests_properties(gemm_async_16b_m_tail_multitile_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=async-16b path=fast-pipeline-16b shape=65x128x32 status=PASS.*reference=cpu"
)

add_test(
    NAME gemm_async_16b_stage_reuse_validate
    COMMAND gemm_runner --kernel async-16b --m 65 --n 128 --k 48 --mode validate
)

set_tests_properties(gemm_async_16b_stage_reuse_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=async-16b path=fast-pipeline-16b shape=65x128x48 status=PASS.*reference=cpu"
)

add_test(
    NAME gemm_async_16b_n_nonmultiple_fallback_validate
    COMMAND gemm_runner --kernel async-16b --m 128 --n 130 --k 128 --mode validate
)

set_tests_properties(gemm_async_16b_n_nonmultiple_fallback_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=async-16b path=fallback-register shape=128x130x128 status=PASS.*reference=cpu"
)

add_test(
    NAME gemm_async_16b_k_nonmultiple_fallback_validate
    COMMAND gemm_runner --kernel async-16b --m 128 --n 128 --k 130 --mode validate
)

set_tests_properties(gemm_async_16b_k_nonmultiple_fallback_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=async-16b path=fallback-register shape=128x128x130 status=PASS.*reference=cpu"
)

add_test(
    NAME gemm_cublas_rectangular_validate
    COMMAND gemm_runner --kernel cublas-fp32 --m 17 --n 19 --k 23 --mode validate
)

set_tests_properties(gemm_cublas_rectangular_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=cublas-fp32 path=cublas-pedantic-fp32 shape=17x19x23 status=PASS.*reference=cpu"
)

add_test(
    NAME gemm_cublas_narrow_rectangular_validate
    COMMAND gemm_runner --kernel cublas-fp32 --m 31 --n 7 --k 13 --mode validate
)

set_tests_properties(gemm_cublas_narrow_rectangular_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=cublas-fp32 path=cublas-pedantic-fp32 shape=31x7x13 status=PASS.*reference=cpu"
)

add_test(
    NAME gemm_register_large_reference_validate
    COMMAND gemm_runner --kernel register --m 512 --n 512 --k 512 --mode validate
)

set_tests_properties(gemm_register_large_reference_validate PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel=register path=register shape=512x512x512 status=PASS.*reference=cublas-pedantic-fp32"
)
