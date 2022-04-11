# Set list of headers
set(HDR
    "nntile.hh"
    "nntile/starpu.hh"
    "nntile/base_type.hh"
    "nntile/tile.hh"
    "nntile/tile/traits.hh"
    "nntile/tile/tile.hh"
    "nntile/tile/randn.hh"
    "nntile/tile/cblas.hh"
    "nntile/tile/gemm.hh"
    "nntile/tile/copy.hh"
    "nntile/tile/bias.hh"
    "nntile/tile/gelu.hh"
    "nntile/tensor.hh"
    "nntile/tensor/traits.hh"
    "nntile/tensor/tensor.hh"
    "nntile/tensor/gemm.hh"
    )

target_sources(nntile PUBLIC ${HDR})

