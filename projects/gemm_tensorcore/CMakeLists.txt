find_package(CUDAToolkit REQUIRED)

add_library(gemm_tensorcore_common STATIC
    common/reference.cu
)

target_include_directories(gemm_tensorcore_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gemm_tensorcore_common
    PUBLIC
        CUDA::cudart
)

target_compile_options(gemm_tensorcore_common
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xcompiler=-Wall,-Wextra>
)

add_library(gemm_tensorcore_wmma_single STATIC
    kernels/wmma_single.cu
)

target_include_directories(gemm_tensorcore_wmma_single
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gemm_tensorcore_wmma_single
    PUBLIC
        gpu_kernel_headers
        CUDA::cudart
)

target_compile_options(gemm_tensorcore_wmma_single
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-warn-spills;-Xcompiler=-Wall,-Wextra>
)

add_executable(gemm_tensorcore_runner
    runner/main.cu
)

target_link_libraries(gemm_tensorcore_runner
    PRIVATE
        gemm_tensorcore_common
        gemm_tensorcore_wmma_single
        gpu_kernel_common
        CUDA::cudart
)

target_compile_options(gemm_tensorcore_runner
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xcompiler=-Wall,-Wextra>
)

if(BUILD_TESTING)
    add_executable(gemm_tensorcore_reference_tests
        tests/reference_tests.cu
    )

    target_link_libraries(gemm_tensorcore_reference_tests
        PRIVATE
            gemm_tensorcore_common
            CUDA::cudart
    )

    target_compile_options(gemm_tensorcore_reference_tests
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>
    )

    add_test(
        NAME gemm_tensorcore_reference_tests
        COMMAND gemm_tensorcore_reference_tests
    )
endif()
