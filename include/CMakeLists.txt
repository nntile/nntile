# @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file include/CMakeLists.txt
# Set NNTile headers
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2022-08-09

# Set list of headers
set(BASE_HDR
    "nntile.hh"
    "nntile/base_types.hh"
    "nntile/constants.hh"
    )

set(KERNEL_HDR
    "nntile/kernel/cpu/bias.hh"
    "nntile/kernel/cpu/copy.hh"
    "nntile/kernel/cpu/gelu.hh"
    "nntile/kernel/cpu/gelutanh.hh"
    "nntile/kernel/cpu/randn.hh"
    "nntile/kernel/cpu/sumnorm.hh"
    "nntile/kernel/cpu/normalize.hh"
    )

set(STARPU_HDR
    "nntile/starpu.hh"
    "nntile/starpu/bias.hh"
    #"nntile/starpu/clear.hh"
    #"nntile/starpu/copy.hh"
    #"nntile/starpu/gelu.hh"
    )

set(TILE_HDR
    #"nntile/tile.hh"
    #"nntile/tile/traits.hh"
    #"nntile/tile/tile.hh"
    #"nntile/tile/randn.hh"
    #"nntile/tile/copy.hh"
    #"nntile/tile/gemm.hh"
    #"nntile/tile/bias.hh"
    #"nntile/tile/gelu.hh"
    #"nntile/tile/relu.hh"
    #"nntile/tile/sumnorm.hh"
    #"nntile/tile/normalize.hh"
    )

set(TENSOR_HDR
    #"nntile/tensor.hh"
    #"nntile/tensor/traits.hh"
    #"nntile/tensor/tensor.hh"
    #"nntile/tensor/randn.hh"
    #"nntile/tensor/copy.hh"
    #"nntile/tensor/gemm.hh"
    #"nntile/tensor/bias.hh"
    #"nntile/tensor/gelu.hh"
    #"nntile/tensor/relu.hh"
    #"nntile/tensor/sumnorm.hh"
    )

set(LAYER_HDR
    )

if(NNTILE_USE_CUDA)
    set(KERNEL_HDR
        ${KERNEL_HDR}
        #"kernel/cuda/gemm.hh"
        #"kernel/cuda/bias.hh"
        )
    set(TILE_HDR
        ${TILE_HDR}
        )
endif()

set(HDR ${BASE_HDR} ${KERNEL_HDR} ${STARPU_HDR} ${TILE_HDR} ${TENSOR_HDR}
    ${LAYER_HDR})

target_sources(nntile PUBLIC ${HDR})

