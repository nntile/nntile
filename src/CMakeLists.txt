# Set list of sources
set(SRC
    "tile/traits.cc"
    "tile/randn.cc"
    "tile/gemm.cc"
    "tile/copy.cc"
    "tile/bias.cc"
    "tile/gelu.cc"
    )

target_sources(nntile PRIVATE ${SRC})

