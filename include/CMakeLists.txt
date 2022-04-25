# Set list of headers
set(HDR
    "nntile.hh"
    "nntile/base_types.hh"
    "nntile/constants.hh"
    "nntile/starpu.hh"
    "nntile/tile.hh"
    "nntile/tile/traits.hh"
    "nntile/tile/tile.hh"
    "nntile/tile/randn.hh"
    "nntile/tile/copy.hh"
    "nntile/tile/gemm.hh"
    "nntile/tile/bias.hh"
    "nntile/tile/gelu.hh"
    "nntile/tile/norm.hh"
    "nntile/tensor.hh"
    "nntile/tensor/traits.hh"
    "nntile/tensor/tensor.hh"
    "nntile/tensor/randn.hh"
    "nntile/tensor/copy.hh"
    "nntile/tensor/gemm.hh"
    "nntile/tensor/bias.hh"
    "nntile/tensor/gelu.hh"
    )

target_sources(nntile PUBLIC ${HDR})

