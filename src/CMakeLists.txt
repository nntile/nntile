# Set list of sources
set(KERNEL_SRC
    "kernel/cpu/bias.cc"
    "kernel/cpu/bias2.cc"
    "kernel/cpu/clear.cc"
    "kernel/cpu/copy.cc"
    "kernel/cpu/gelu.cc"
    "kernel/cpu/scal.cc"
    "kernel/cpu/sumnorm.cc"
    )

set(TILE_SRC
    "tile/traits.cc"
    "tile/randn.cc"
    "tile/clear.cc"
    "tile/copy.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/tile/gemm.cc"
    "tile/bias.cc"
    "tile/bias2.cc"
    "tile/gelu.cc"
    "tile/relu.cc"
    "tile/norm.cc"
    "tile/scal.cc"
    "tile/sumnorm.cc"
    )

set(TENSOR_SRC
    "tensor/traits.cc"
    "tensor/randn.cc"
    "tensor/clear.cc"
    "tensor/copy.cc"
    "tensor/gemm.cc"
    "tensor/bias.cc"
    "tensor/bias2.cc"
    "tensor/gelu.cc"
    "tensor/relu.cc"
    "tensor/norm.cc"
    "tensor/scal.cc"
    "tensor/sumnorm.cc"
    )

if(NNTILE_USE_CBLAS)
    # Configure src/kernel/cpu/gemm.cc.in that relies on CBLAS_H_NAME and
    # CBLAS_INT_TYPE
    configure_file("kernel/cpu/gemm.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/kernel/cpu/gemm.cc" @ONLY)
    set(KERNEL_SRC
        ${KERNEL_SRC}
        "${CMAKE_CURRENT_BINARY_DIR}/kernel/cpu/gemm.cc"
        )
endif()

if(NNTILE_USE_CUDA)
    set(KERNEL_SRC
        ${KERNEL_SRC}
        "kernel/cuda/gemm.cc"
        "kernel/cuda/bias.cu"
        "kernel/cuda/bias2.cu"
        )
    set(TILE_SRC
        ${TILE_SRC}
	"tile/relu.cu"
	"tile/norm.cu"
	"tile/bias.cu"
        )
endif()

target_sources(nntile PRIVATE ${KERNEL_SRC} ${TILE_SRC} ${TENSOR_SRC})

