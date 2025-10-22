/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/flash_sdpa_fwd_cudnn/cuda.hh
 * Flash attention scaled dot-product attention forward pass using cuDNN
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>
#include <cudnn.h>

namespace nntile::kernel::flash_sdpa_fwd_cudnn
{

//! Flash attention forward pass using cuDNN
template<typename T>
void cuda(
    cudaStream_t stream,
    Index seq,
    Index head,
    Index batch,
    const T *K,
    const T *Q,
    const T *mask,
    T *logsumexp,
    const T *V,
    T *A
) noexcept;

} // namespace nntile::kernel::flash_sdpa_fwd_cudnn
