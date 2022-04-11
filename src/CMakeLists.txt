# Set list of sources
set(SRC
    "tile/traits.cc"
    "tile/randn.cc"
    "tile/gemm.cc"
    "tile/copy.cc"
    )

target_sources(nntile PRIVATE ${SRC})

