find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(attention_prefill_fingerprint_script
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/source_fingerprint.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${attention_prefill_fingerprint_script}")
file(GLOB_RECURSE attention_prefill_evidence_source_candidates
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    "${PROJECT_SOURCE_DIR}/common/include/*"
    "${PROJECT_SOURCE_DIR}/common/src/*"
    "${PROJECT_SOURCE_DIR}/projects/flash_attention/include/*"
    "${PROJECT_SOURCE_DIR}/projects/flash_attention/kernels/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/evidence/*"
)
execute_process(
    COMMAND ${Python3_EXECUTABLE} "${attention_prefill_fingerprint_script}"
            --repo-root "${PROJECT_SOURCE_DIR}" --print-files
    OUTPUT_VARIABLE attention_prefill_evidence_source_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "\n" ";" attention_prefill_evidence_sources
    "${attention_prefill_evidence_source_output}")
foreach(relative_path IN LISTS attention_prefill_evidence_sources)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/${relative_path}")
endforeach()
execute_process(
    COMMAND ${Python3_EXECUTABLE} "${attention_prefill_fingerprint_script}"
            --repo-root "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE ATTENTION_PREFILL_EVIDENCE_SOURCE_SHA256
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
if(NOT ATTENTION_PREFILL_EVIDENCE_SOURCE_SHA256 MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "Evidence source fingerprint is invalid")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" attention_prefill_build_type)
list(JOIN CMAKE_CUDA_ARCHITECTURES "-" attention_prefill_cuda_architectures)
file(REAL_PATH "${CMAKE_CUDA_COMPILER}"
    attention_prefill_cuda_compiler_realpath)
set(attention_prefill_kernel_cuda_options
    "-O3;-lineinfo;-Xptxas=-warn-spills;-Xcompiler=-Wall,-Wextra")
set(attention_prefill_evidence_cuda_options
    "-O3;-lineinfo;-Xcompiler=-Wall,-Wextra")
get_target_property(attention_prefill_flash_cuda_options
    flash_attention_kernels GPU_KERNEL_EVIDENCE_CUDA_OPTIONS)
if(NOT attention_prefill_flash_cuda_options)
    message(FATAL_ERROR "Unable to read flash_attention_kernels compile options")
endif()
string(CONCAT ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT_PAYLOAD
    "schema_version=1\n"
    "build_type=${CMAKE_BUILD_TYPE}\n"
    "cuda_architectures=${CMAKE_CUDA_ARCHITECTURES}\n"
    "cuda_compiler_id=${CMAKE_CUDA_COMPILER_ID}\n"
    "cuda_compiler_version=${CMAKE_CUDA_COMPILER_VERSION}\n"
    "cuda_compiler_realpath=${attention_prefill_cuda_compiler_realpath}\n"
    "cmake_cuda_flags=${CMAKE_CUDA_FLAGS}\n"
    "cmake_cuda_flags_release=${CMAKE_CUDA_FLAGS_RELEASE}\n"
    "target_compile_options_evidence_support=${attention_prefill_evidence_cuda_options}\n"
    "target_compile_options_attention_kernel=${attention_prefill_kernel_cuda_options}\n"
    "target_compile_options_flash_kernels=${attention_prefill_flash_cuda_options}\n"
)
string(SHA256 ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT_PAYLOAD_SHA256
    "${ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT_PAYLOAD}")
string(SUBSTRING "${ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT_PAYLOAD_SHA256}"
    0 16 attention_prefill_build_contract_short_hash)
set(ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT
    "${attention_prefill_build_type}-sm${attention_prefill_cuda_architectures}-${attention_prefill_build_contract_short_hash}")
if(NOT ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT MATCHES "^[A-Za-z0-9._-]+$")
    message(FATAL_ERROR "Evidence build contract is not a safe token")
endif()

set(ATTENTION_PREFILL_EVIDENCE_BUILD_ATTESTATION
    "${CMAKE_CURRENT_BINARY_DIR}/attention_prefill_build_attestation.txt")
file(WRITE "${ATTENTION_PREFILL_EVIDENCE_BUILD_ATTESTATION}"
    "${ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT_PAYLOAD}"
    "build_contract=${ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT}\n"
    "build_contract_payload_sha256=${ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT_PAYLOAD_SHA256}\n")

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
        $<$<COMPILE_LANGUAGE:CUDA>:${attention_prefill_kernel_cuda_options}>
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

add_library(attention_prefill_evidence_support STATIC
    evidence/runner.cu
)

target_include_directories(attention_prefill_evidence_support
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(attention_prefill_evidence_support
    PUBLIC
        attention_prefill_kernel
        flash_attention_kernels
        gpu_kernel_common
        CUDA::cudart
)

target_compile_options(attention_prefill_evidence_support
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${attention_prefill_evidence_cuda_options}>
)

target_compile_definitions(attention_prefill_evidence_support
    PUBLIC
        ATTENTION_PREFILL_EVIDENCE_SOURCE_SHA256="${ATTENTION_PREFILL_EVIDENCE_SOURCE_SHA256}"
        ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT="${ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT}"
        ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT_PAYLOAD_SHA256="${ATTENTION_PREFILL_EVIDENCE_BUILD_CONTRACT_PAYLOAD_SHA256}"
)

add_executable(attention_prefill_evidence_runner
    evidence/main.cu
)

target_link_libraries(attention_prefill_evidence_runner
    PRIVATE
        attention_prefill_evidence_support
)

add_custom_command(TARGET attention_prefill_evidence_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E touch
            "${ATTENTION_PREFILL_EVIDENCE_BUILD_ATTESTATION}"
    VERBATIM
)

if(BUILD_TESTING)

    add_test(
        NAME attention_prefill_source_fingerprint_tests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/source_fingerprint_test.py
                --repo-root ${PROJECT_SOURCE_DIR}
                --runner $<TARGET_FILE:attention_prefill_evidence_runner>
    )

    add_test(
        NAME attention_prefill_render_m1_results_tests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/render_m1_results_test.py
    )

    add_test(
        NAME attention_prefill_benchmark_m1_tests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_m1_test.py
    )

    add_test(
        NAME attention_prefill_summarize_m1_ncu_tests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/summarize_m1_ncu_test.py
    )

    add_test(
        NAME attention_prefill_m1_profile_sass_tests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/m1_profile_sass_test.py
    )

    add_test(
        NAME attention_prefill_evidence_br1_validate_128x64_noncausal
        COMMAND attention_prefill_evidence_runner
                --implementation br1 --n 128 --d 64 --causal 0
                --mode validate --warmup 0 --iterations 1
    )
    set_tests_properties(
        attention_prefill_evidence_br1_validate_128x64_noncausal
        PROPERTIES PASS_REGULAR_EXPRESSION
            "implementation=br1 path=br1 shape=128x64 causal=0 .*status=PASS"
    )

    add_test(
        NAME attention_prefill_evidence_br4_validate_128x64_noncausal
        COMMAND attention_prefill_evidence_runner
                --implementation br4 --n 128 --d 64 --causal 0
                --mode validate --warmup 0 --iterations 1
    )
    set_tests_properties(
        attention_prefill_evidence_br4_validate_128x64_noncausal
        PROPERTIES PASS_REGULAR_EXPRESSION
            "implementation=br4 path=br4 shape=128x64 causal=0 .*status=PASS"
    )

    add_test(
        NAME attention_prefill_evidence_br1_benchmark_128x64_noncausal
        COMMAND attention_prefill_evidence_runner
                --implementation br1 --n 128 --d 64 --causal 0
                --mode benchmark --warmup 0 --iterations 1
    )
    set_tests_properties(
        attention_prefill_evidence_br1_benchmark_128x64_noncausal
        PROPERTIES PASS_REGULAR_EXPRESSION
            "implementation=br1 path=br1 shape=128x64 causal=0 .*status=PASS"
    )

    add_test(
        NAME attention_prefill_evidence_runner_integration_tests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/evidence_runner_integration_test.py
                --runner $<TARGET_FILE:attention_prefill_evidence_runner>
    )

    add_executable(attention_prefill_evidence_runner_tests
        tests/evidence_runner_tests.cpp
    )

    target_link_libraries(attention_prefill_evidence_runner_tests
        PRIVATE
            attention_prefill_evidence_support
    )

    target_compile_options(attention_prefill_evidence_runner_tests
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )

    add_test(
        NAME attention_prefill_evidence_runner_tests
        COMMAND attention_prefill_evidence_runner_tests
    )

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
