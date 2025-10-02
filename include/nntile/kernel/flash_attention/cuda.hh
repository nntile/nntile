/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/flash_attention/cuda.hh
 * Flash attention forward pass on CUDA using cuDNN
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::flash_attention
{

template<typename T>
void cuda(cudaStream_t stream, Index batch, Index num_heads, Index seq_len,
        Index head_dim, const T *Q, const T *K, const T *V, Scalar scale, T *O)
    noexcept;

} // namespace nntile::kernel::flash_attention
