# Set list of sources
set(SRC
    "tile/traits.cc"
    "tile/randn.cc"
    "tile/copy.cc"
    "tile/gemm.cc"
    "tile/bias.cc"
    "tile/gelu.cc"
    "tensor/traits.cc"
    "tensor/randn.cc"
    "tensor/copy.cc"
    "tensor/gemm.cc"
    "tensor/bias.cc"
    "tensor/gelu.cc"
    )

target_sources(nntile PRIVATE ${SRC})

