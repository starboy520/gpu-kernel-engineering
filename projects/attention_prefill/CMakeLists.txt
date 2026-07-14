find_package(CUDAToolkit REQUIRED)

add_library(attention_prefill_kernel STATIC
    kernels/query_tiled.cu
)

target_include_directories(attention_prefill_kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(attention_prefill_kernel
    PUBLIC
        gpu_kernel_headers
        CUDA::cudart
)

target_compile_options(attention_prefill_kernel
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-warn-spills;-Xcompiler=-Wall,-Wextra>
)

add_executable(attention_prefill_runner
    runner/main.cu
)

target_link_libraries(attention_prefill_runner
    PRIVATE
        attention_prefill_kernel
        gpu_kernel_common
        CUDA::cudart
)

target_compile_options(attention_prefill_runner
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xcompiler=-Wall,-Wextra>
)

if(BUILD_TESTING)
    add_executable(attention_prefill_contract_tests
        tests/problem_contract_tests.cu
    )

    target_link_libraries(attention_prefill_contract_tests
        PRIVATE
            attention_prefill_kernel
            CUDA::cudart
    )

    add_test(
        NAME attention_prefill_problem_contract_tests
        COMMAND attention_prefill_contract_tests
    )

    add_test(
        NAME attention_prefill_query_tiled_validate
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_query_tiled.sh
                $<TARGET_FILE:attention_prefill_runner>
    )
    set_tests_properties(attention_prefill_query_tiled_validate PROPERTIES
        PASS_REGULAR_EXPRESSION
            "\\[test_query_tiled\\] summary shape_executions=126 special_executions=2 pass=128 status=PASS"
    )
endif()
