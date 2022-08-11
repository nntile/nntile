# @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file src/CMakeLists.txt
# Set NNTile sources to compile
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2022-08-10

# Set list of sources
set(KERNEL_SRC
    "kernel/cpu/bias.cc"
    "kernel/cpu/copy.cc"
    "kernel/cpu/gelu.cc"
    "kernel/cpu/gelutanh.cc"
    "kernel/cpu/normalize.cc"
    "kernel/cpu/randn.cc"
    "kernel/cpu/sumnorm.cc"
    )

set(STARPU_SRC
    "starpu/bias.cc"
    "starpu/clear.cc"
    "starpu/copy.cc"
    "starpu/gelu.cc"
    "starpu/gelutanh.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/starpu/gemm.cc"
    "starpu/normalize.cc"
    "starpu/randn.cc"
    "starpu/sumnorm.cc"
    )

set(TILE_SRC
    #"tile/traits.cc"
    #"tile/randn.cc"
    #"tile/clear.cc"
    #"tile/copy.cc"
    #"tile/gemm.cc"
    #"tile/bias.cc"
    #"tile/gelu.cc"
    #"tile/relu.cc"
    #"tile/sumnorm.cc"
    #"tile/normalize.cc"
    )

set(TENSOR_SRC
    #"tensor/traits.cc"
    #"tensor/randn.cc"
    #"tensor/clear.cc"
    #"tensor/copy.cc"
    #"tensor/gemm.cc"
    #"tensor/bias.cc"
    #"tensor/gelu.cc"
    #"tensor/relu.cc"
    #"tensor/sumnorm.cc"
    )

set(LAYER_SRC
    )

if(NNTILE_USE_CUDA)
    set(KERNEL_SRC
        ${KERNEL_SRC}
        #"kernel/cuda/gemm.cc"
        #"kernel/cuda/bias.cu"
        )
    set(TILE_SRC
        ${TILE_SRC}
        #"tile/relu.cu"
        #"tile/norm.cu"
        #"tile/bias.cu"
        )
endif()

set(SRC ${KERNEL_SRC} ${STARPU_SRC} ${TILE_SRC} ${TENSOR_SRC} ${LAYER_SRC})

target_sources(nntile PRIVATE ${SRC})

