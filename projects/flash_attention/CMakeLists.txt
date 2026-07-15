find_package(CUDAToolkit REQUIRED)

add_library(flash_attention_common STATIC
    common/reference.cpp
)

target_include_directories(flash_attention_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(flash_attention_common
    PUBLIC
    gpu_kernel_common
        CUDA::cudart
)

target_compile_options(flash_attention_common
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_library(flash_attention_kernels STATIC
    kernels/naive.cu
    kernels/tiled.cu
    kernels/tiled_parallel.cu
    kernels/tiled_async.cu
)

set(flash_attention_kernel_cuda_options
    "-O3;-lineinfo;-Xptxas=-warn-spills;-Xcompiler=-Wall,-Wextra")
set_property(TARGET flash_attention_kernels PROPERTY
    GPU_KERNEL_EVIDENCE_CUDA_OPTIONS "${flash_attention_kernel_cuda_options}")

target_include_directories(flash_attention_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(flash_attention_kernels
    PUBLIC
        gpu_kernel_headers
        CUDA::cudart
)

target_compile_options(flash_attention_kernels
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${flash_attention_kernel_cuda_options}>
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

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

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

    add_test(
        NAME flash_attention_render_results_tests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/render_results_test.py
    )

    add_test(
        NAME flash_attention_summarize_ncu_tests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/summarize_ncu_test.py
    )

    function(add_flash_attention_naive_validation_test test_name n d causal input_pattern)
        add_test(
            NAME ${test_name}
            COMMAND flash_attention_runner
                --kernel naive
                --n ${n}
                --d ${d}
                --causal ${causal}
                --input-pattern ${input_pattern}
                --mode validate
                --warmup 0
                --iterations 1
        )
        set_tests_properties(${test_name} PROPERTIES
            PASS_REGULAR_EXPRESSION
                "kernel=naive path=naive shape=${n}x${d} causal=${causal} input_pattern=${input_pattern} status=PASS"
        )
    endfunction()

    add_flash_attention_naive_validation_test(
        flash_attention_naive_minimal_noncausal_validate 1 1 0 random
    )
    add_flash_attention_naive_validation_test(
        flash_attention_naive_small_causal_validate 3 2 1 random
    )
    add_flash_attention_naive_validation_test(
        flash_attention_naive_tail_noncausal_validate 37 24 0 random
    )
    add_flash_attention_naive_validation_test(
        flash_attention_naive_tail_causal_validate 37 24 1 random
    )
    add_flash_attention_naive_validation_test(
        flash_attention_naive_negative_scores_validate 37 24 0 negative-scores
    )
    add_flash_attention_naive_validation_test(
        flash_attention_naive_zero_qk_causal_validate 37 24 1 zero-qk
    )

    add_test(
        NAME flash_attention_tiled_validate
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_tiled.sh
                $<TARGET_FILE:flash_attention_runner>
    )
    set_tests_properties(flash_attention_tiled_validate PROPERTIES
        PASS_REGULAR_EXPRESSION
            "\\[test_tiled\\] summary executions=13 pass=13 status=PASS"
    )

    add_test(
        NAME flash_attention_tiled_parallel_validate
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_tiled_parallel.sh
                $<TARGET_FILE:flash_attention_runner>
    )
    set_tests_properties(flash_attention_tiled_parallel_validate PROPERTIES
        PASS_REGULAR_EXPRESSION
            "\\[test_tiled_parallel\\] summary executions=13 pass=13 status=PASS"
    )

    add_test(
        NAME flash_attention_tiled_async_validate
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_tiled_async.sh
                $<TARGET_FILE:flash_attention_runner>
    )
    set_tests_properties(flash_attention_tiled_async_validate PROPERTIES
        PASS_REGULAR_EXPRESSION
            "\\[test_tiled_async\\] summary executions=13 pass=13 status=PASS"
    )
endif()