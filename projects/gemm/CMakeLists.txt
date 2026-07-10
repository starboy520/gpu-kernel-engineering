add_custom_target(gemm_scaffold ALL
    COMMAND ${CMAKE_COMMAND} -E echo "GEMM scaffold configured for sm_${CMAKE_CUDA_ARCHITECTURES}"
)