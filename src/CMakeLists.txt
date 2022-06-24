# Set list of sources
set(SRC
    "tile/traits.cc"
    "tile/randn.cc"
    "tile/copy.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/tile/gemm.cc"
    "tile/bias.cc"
    "tile/gelu.cc"
    "tile/relu.cc"
    "tile/norm.cc"
    "tensor/traits.cc"
    "tensor/randn.cc"
    "tensor/copy.cc"
    "tensor/gemm.cc"
    "tensor/bias.cc"
    "tensor/gelu.cc"
    "tensor/relu.cc"
    "tensor/norm.cc"
    )

if(USE_CUDA)
    set(SRC
        ${SRC}
	#"tile/relu.cu"
        )
endif()

target_sources(nntile PRIVATE ${SRC})

