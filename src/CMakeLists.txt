# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
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
# @author Aleksandr Katrutsa
# @date 2023-02-10

# Set list of sources
set(KERNEL_SRC
    "kernel/bias/cpu.cc"
    "kernel/dgelu/cpu.cc"
    "kernel/dgelutanh/cpu.cc"
    "kernel/drelu/cpu.cc"
    "kernel/gelu/cpu.cc"
    "kernel/gelutanh/cpu.cc"
    "kernel/hypot/cpu.cc"
    "kernel/normalize/cpu.cc"
    "kernel/prod/cpu.cc"
    "kernel/randn/cpu.cc"
    "kernel/relu/cpu.cc"
    "kernel/subcopy/cpu.cc"
    "kernel/sumnorm/cpu.cc"
    "kernel/maxsumexp/cpu.cc"
    "kernel/softmax/cpu.cc"
    "kernel/maximum/cpu.cc"
    )

if(NNTILE_USE_CUDA)
    set(KERNEL_SRC
        ${KERNEL_SRC}
        "kernel/bias/cuda.cu"
        "kernel/dgelu/cuda.cu"
        "kernel/dgelutanh/cuda.cu"
        "kernel/drelu/cuda.cu"
        "kernel/gelu/cuda.cu"
        "kernel/gelutanh/cuda.cu"
        "kernel/normalize/cuda.cu"
        "kernel/prod/cuda.cu"
        "kernel/relu/cuda.cu"
        "kernel/sumnorm/cuda.cu"
        "kernel/maxsumexp/cuda.cu"
        "kernel/softmax/cuda.cu"
        )
endif()

set(STARPU_SRC
    "starpu/bias.cc"
    "starpu/clear.cc"
    "starpu/dgelu.cc"
    "starpu/dgelutanh.cc"
    "starpu/drelu.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/starpu/axpy.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/starpu/gemm.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/starpu/nrm2.cc"
    "starpu/gelu.cc"
    "starpu/gelutanh.cc"
    "starpu/hypot.cc"
    "starpu/normalize.cc"
    "starpu/prod.cc"
    "starpu/randn.cc"
    "starpu/relu.cc"
    "starpu/subcopy.cc"
    "starpu/sumnorm.cc"
    "starpu/maxsumexp.cc"
    "starpu/softmax.cc"
    "starpu/maximum.cc"
    )

set(TILE_SRC
    "tile/traits.cc"
    "tile/axpy.cc"
    "tile/bias.cc"
    "tile/clear.cc"
    "tile/copy.cc"
    "tile/copy_intersection.cc"
    "tile/dgelu.cc"
    "tile/dgelutanh.cc"
    "tile/drelu.cc"
    "tile/gemm.cc"
    "tile/gelu.cc"
    "tile/gelutanh.cc"
    "tile/nrm2.cc"
    "tile/normalize.cc"
    "tile/prod.cc"
    "tile/randn.cc"
    "tile/relu.cc"
    "tile/sumnorm.cc"
    "tile/maxsumexp.cc"
    "tile/softmax.cc"
    "tile/maximum.cc"
    )

set(TENSOR_SRC
    "tensor/traits.cc"
    "tensor/distributions.cc"
    "tensor/axpy.cc"
    "tensor/bias.cc"
    "tensor/clear.cc"
    "tensor/copy.cc"
    "tensor/copy_intersection.cc"
    "tensor/dgelu.cc"
    "tensor/dgelutanh.cc"
    "tensor/drelu.cc"
    "tensor/gather.cc"
    "tensor/gemm.cc"
    "tensor/gelu.cc"
    "tensor/gelutanh.cc"
    "tensor/nrm2.cc"
    "tensor/normalize.cc"
    "tensor/prod.cc"
    "tensor/randn.cc"
    "tensor/relu.cc"
    "tensor/scatter.cc"
    "tensor/sumnorm.cc"
    "tensor/maxsumexp.cc"
    "tensor/softmax.cc"
    "tensor/maximum.cc"
    )

set(LAYER_SRC
    "layer/gelu.cc"
    #"layer/gelutanh.cc"
    "layer/linear.cc"
    #"layer/mlp.cc"
    )

set(MODEL_SRC
    "model/deep_linear.cc"
    )

set(OPTIMIZER_SRC
    #"optimizer/grad_descent.cc"
    )

set(SRC ${KERNEL_SRC} ${STARPU_SRC} ${TILE_SRC} ${TENSOR_SRC} ${LAYER_SRC})
    #${MODEL_SRC} ${OPTIMIZER_SRC})

target_sources(nntile PRIVATE ${SRC})

