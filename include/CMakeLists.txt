# Set list of headers
set(KERNEL_HDR
    "nntile/kernel/cpu/bias.hh"
    "nntile/kernel/cpu/bias2.hh"
    "nntile/kernel/cpu/copy.hh"
    "nntile/kernel/cpu/gelu.hh"
    "nntile/kernel/cpu/scal.hh"
    "nntile/kernel/cpu/sumnorm.hh"
    )

set(TILE_HDR
    "nntile/tile.hh"
    "nntile/tile/traits.hh"
    "nntile/tile/tile.hh"
    "nntile/tile/randn.hh"
    "nntile/tile/copy.hh"
    "nntile/tile/gemm.hh"
    "nntile/tile/bias.hh"
    "nntile/tile/bias2.hh"
    "nntile/tile/gelu.hh"
    "nntile/tile/relu.hh"
    "nntile/tile/norm.hh"
    "nntile/tile/scal.hh"
    "nntile/tile/sumnorm.hh"
    )

set(TENSOR_SRC
    "nntile/tensor.hh"
    "nntile/tensor/traits.hh"
    "nntile/tensor/tensor.hh"
    "nntile/tensor/randn.hh"
    "nntile/tensor/copy.hh"
    "nntile/tensor/gemm.hh"
    "nntile/tensor/bias.hh"
    "nntile/tensor/bias2.hh"
    "nntile/tensor/gelu.hh"
    "nntile/tensor/relu.hh"
    "nntile/tensor/norm.hh"
    "nntile/tensor/scal.hh"
    "nntile/tensor/sumnorm.hh"
    )

if(NNTILE_USE_CBLAS)
    set(KERNEL_HDR
        ${KERNEL_HDR}
        "nntile/kernel/cpu/gemm.hh"
        )
endif()

if(NNTILE_USE_CUDA)
    set(KERNEL_HDR
        ${KERNEL_HDR}
        "kernel/cuda/gemm.hh"
        "kernel/cuda/bias.hh"
        "kernel/cuda/bias2.hh"
        )
    set(TILE_HDR
        ${TILE_HDR}
        )
endif()

set(HDR
    "nntile.hh"
    "nntile/base_types.hh"
    "nntile/constants.hh"
    "nntile/starpu.hh"
    # Kernel operations
    ${KERNEL_HDR}
    # Tile operations
    ${TILE_HDR}
    # Tensor operations
    ${TENSOR_HDR}
    # Layers
    "nntile/layer.hh"
    "nntile/layer/fully_connected.hh"
    "nntile/layer/mixer.hh"
    )

target_sources(nntile PUBLIC ${HDR})

