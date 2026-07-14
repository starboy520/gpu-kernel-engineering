find_package(CUDAToolkit REQUIRED)

add_library(gpu_kernel_headers INTERFACE)

target_include_directories(gpu_kernel_headers
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gpu_kernel_headers
    INTERFACE
        CUDA::cudart
)

add_library(gpu_kernel_common STATIC
    validation.cpp
)

target_link_libraries(gpu_kernel_common
    PUBLIC
        gpu_kernel_headers
)

target_compile_options(gpu_kernel_common
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_executable(gpu_kernel_common_tests
    tests/common_tests.cpp
)

target_link_libraries(gpu_kernel_common_tests
    PRIVATE
        gpu_kernel_common
)

target_compile_options(gpu_kernel_common_tests
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_test(NAME gpu_kernel_common_tests COMMAND gpu_kernel_common_tests)

add_test(
    NAME gpu_kernel_common_script_tests
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/common_test.sh
)