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
# @date 2022-08-31

# Set list of sources
set(KERNEL_SRC
    "kernel/bias/cpu.cc"
    "kernel/copy/cpu.cc"
    "kernel/gelu/cpu.cc"
    "kernel/gelutanh/cpu.cc"
    "kernel/normalize/cpu.cc"
    "kernel/randn/cpu.cc"
    "kernel/sumnorm/cpu.cc"
    "kernel/relu/cpu.cc"
    )

if(NNTILE_USE_CUDA)
    set(KERNEL_SRC
        ${KERNEL_SRC}
        "kernel/bias/cuda.cu"
        "kernel/gelu/cuda.cu"
        "kernel/gelutanh/cuda.cu"
        "kernel/normalize/cuda.cu"
        "kernel/relu/cuda.cu"
        "kernel/sumnorm/cuda.cu"
        )
endif()

set(STARPU_SRC
    "starpu/bias.cc"
    "starpu/clear.cc"
    "starpu/copy.cc"
    "starpu/gelu.cc"
    "starpu/gelutanh.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/starpu/gemm.cc"
    "starpu/normalize.cc"
    "starpu/randn.cc"
    "starpu/relu.cc"
    "starpu/sumnorm.cc"
    )

set(TILE_SRC
    "tile/traits.cc"
    "tile/bias.cc"
    "tile/clear.cc"
    "tile/copy.cc"
    "tile/gemm.cc"
    "tile/gelu.cc"
    "tile/gelutanh.cc"
    "tile/randn.cc"
    "tile/relu.cc"
    "tile/sumnorm.cc"
    "tile/normalize.cc"
    )

set(TENSOR_SRC
    #"tensor/traits.cc"
    #"tensor/distributions.cc"
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

set(SRC ${KERNEL_SRC} ${STARPU_SRC} ${TILE_SRC} ${TENSOR_SRC} ${LAYER_SRC})

target_sources(nntile PRIVATE ${SRC})

