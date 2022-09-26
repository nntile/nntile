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
# @date 2022-09-19

# Set list of headers
set(BASE_HDR
    "nntile.hh"
    "nntile/base_types.hh"
    "nntile/constants.hh"
    "nntile/config.hh"
    )

set(KERNEL_HDR
    "nntile/kernel/bias.hh"
    "nntile/kernel/bias/cpu.hh"
    "nntile/kernel/gelu.hh"
    "nntile/kernel/gelu/cpu.hh"
    "nntile/kernel/subcopy/cpu.hh"
    "nntile/kernel/gelutanh/cpu.hh"
    "nntile/kernel/normalize/cpu.hh"
    "nntile/kernel/randn/cpu.hh"
    "nntile/kernel/relu/cpu.hh"
    "nntile/kernel/sumnorm/cpu.hh"
    )

if(NNTILE_USE_CUDA)
    set(KERNEL_HDR
        ${KERNEL_HDR}
        "nntile/kernel/bias/cuda.hh"
        "nntile/kernel/gelu/cuda.hh"
        "nntile/kernel/gelutanh/cuda.hh"
        "nntile/kernel/normalize/cuda.hh"
        "nntile/kernel/relu/cuda.hh"
        "nntile/kernel/sumnorm/cuda.hh"
        )
endif()

set(STARPU_HDR
    "nntile/starpu/config.hh"
    "nntile/starpu/bias.hh"
    "nntile/starpu/clear.hh"
    "nntile/starpu/subcopy.hh"
    "nntile/starpu/gelu.hh"
    "nntile/starpu/gelutanh.hh"
    "nntile/starpu/gemm.hh"
    "nntile/starpu/normalize.hh"
    "nntile/starpu/randn.hh"
    "nntile/starpu/relu.hh"
    "nntile/starpu/sumnorm.hh"
    )

set(TILE_HDR
    "nntile/tile.hh"
    "nntile/tile/traits.hh"
    "nntile/tile/tile.hh"
    "nntile/tile/bias.hh"
    "nntile/tile/copy.hh"
    "nntile/tile/copy_intersection.hh"
    "nntile/tile/gemm.hh"
    "nntile/tile/gelu.hh"
    "nntile/tile/gelutanh.hh"
    "nntile/tile/randn.hh"
    "nntile/tile/relu.hh"
    "nntile/tile/sumnorm.hh"
    "nntile/tile/normalize.hh"
    )

set(TENSOR_HDR
    "nntile/tensor.hh"
    "nntile/tensor/traits.hh"
    "nntile/tensor/distributions.hh"
    "nntile/tensor/tensor.hh"
    "nntile/tensor/copy.hh"
    "nntile/tensor/copy_intersection.hh"
    "nntile/tensor/scatter.hh"
    "nntile/tensor/gather.hh"
    "nntile/tensor/clear.hh"
    "nntile/tensor/bias.hh"
    "nntile/tensor/randn.hh"
    "nntile/tensor/gemm.hh"
    "nntile/tensor/gelu.hh"
    "nntile/tensor/gelutanh.hh"
    "nntile/tensor/relu.hh"
    "nntile/tensor/sumnorm.hh"
    "nntile/tensor/normalize.hh"
    )

set(LAYER_HDR
    )

set(HDR ${BASE_HDR} ${KERNEL_HDR} ${STARPU_HDR} ${TILE_HDR} ${TENSOR_HDR})# ${LAYER_HDR})

target_sources(nntile PUBLIC ${HDR})

