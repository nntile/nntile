# Set list of headers
set(HDR
    "nntile.hh"
    "nntile/base_type.hh"
    "nntile/tile.hh"
    "nntile/tile/traits.hh"
    "nntile/tile/tile.hh"
    "nntile/tile/gemm.hh"
    "nntile/tile/gelu.hh"
    )

target_sources(nntile INTERFACE ${HDR})

