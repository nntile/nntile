# Set list of sources
set(KERNEL_SRC
    "kernel/cpu/bias.cc"
    "kernel/cpu/copy.cc"
    "kernel/cpu/gelu.cc"
    "kernel/cpu/gelutanh.cc"
    "kernel/cpu/scal.cc"
    "kernel/cpu/sumnorm.cc"
    "kernel/cpu/normalize.cc"
    )

set(STARPU_SRC
    "starpu/bias.cc"
    "starpu/clear.cc"
    "starpu/copy.cc"
    "starpu/gelu.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/starpu/gemm.cc"
    )

set(TILE_SRC
    "tile/traits.cc"
    "tile/randn.cc"
    "tile/clear.cc"
    "tile/copy.cc"
    "tile/gemm.cc"
    "tile/bias.cc"
    "tile/gelu.cc"
    "tile/relu.cc"
    "tile/scal.cc"
    "tile/sumnorm.cc"
    "tile/normalize.cc"
    )

set(TENSOR_SRC
    "tensor/traits.cc"
    "tensor/randn.cc"
    "tensor/clear.cc"
    "tensor/copy.cc"
    "tensor/gemm.cc"
    "tensor/bias.cc"
    "tensor/gelu.cc"
    "tensor/relu.cc"
    "tensor/scal.cc"
    "tensor/sumnorm.cc"
    )

if(NNTILE_USE_CUDA)
    set(KERNEL_SRC
        ${KERNEL_SRC}
        "kernel/cuda/gemm.cc"
        "kernel/cuda/bias.cu"
        )
    set(TILE_SRC
        ${TILE_SRC}
	"tile/relu.cu"
	"tile/norm.cu"
	"tile/bias.cu"
        )
endif()

target_sources(nntile PRIVATE ${KERNEL_SRC} ${STARPU_SRC} ${TILE_SRC}
    ${TENSOR_SRC})

