find_package(CUDAToolkit REQUIRED)

add_library(flash_attention_common STATIC
    common/reference.cpp
    common/validation.cpp
)

target_include_directories(flash_attention_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(flash_attention_common
    PUBLIC
        CUDA::cudart
)

target_compile_options(flash_attention_common
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_library(flash_attention_kernels STATIC
    kernels/naive.cu
)

target_include_directories(flash_attention_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(flash_attention_kernels
    PUBLIC
        CUDA::cudart
)

target_compile_options(flash_attention_kernels
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-warn-spills;-Xcompiler=-Wall,-Wextra>
)

add_library(flash_attention_runner_support STATIC
    runner/registry.cu
    runner/runner.cu
)

target_include_directories(flash_attention_runner_support
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(flash_attention_runner_support
    PUBLIC
        flash_attention_common
        CUDA::cudart
)

target_compile_options(flash_attention_runner_support
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xcompiler=-Wall,-Wextra>
)

add_executable(flash_attention_runner
    runner/main.cu
)

target_link_libraries(flash_attention_runner
    PRIVATE
        flash_attention_runner_support
        flash_attention_kernels
)

add_executable(flash_attention_common_tests
    tests/common_tests.cpp
)

target_link_libraries(flash_attention_common_tests
    PRIVATE
        flash_attention_common
)

target_compile_options(flash_attention_common_tests
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_test(NAME flash_attention_common_tests COMMAND flash_attention_common_tests)

add_executable(flash_attention_runner_tests
    tests/runner_tests.cpp
)

target_link_libraries(flash_attention_runner_tests
    PRIVATE
        flash_attention_runner_support
        flash_attention_kernels
)

target_compile_options(flash_attention_runner_tests
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_test(NAME flash_attention_runner_tests COMMAND flash_attention_runner_tests)