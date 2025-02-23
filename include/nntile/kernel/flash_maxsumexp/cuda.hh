/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/flash_maxsumexp/cuda.hh
 * CUDA kernel to compute maxsumexp((QK')/sqrt(d)) with masking
 *
 * @version 1.1.0
 * */

#pragma once

#include <cuda_runtime.h>
#include <nntile/base_types.hh>

namespace nntile::kernel::flash_maxsumexp
{

/**
 * CUDA kernel to compute maximum and sum of exponentials for Flash Attention
 * Fuses the following operations:
 * 1. Scaled dot product: tmp = (K'@Q)/sqrt(head)
 * 2. Masking: Apply mask values (-inf) to tmp where mask is 0
 * 3. MaxSumExp: Compute max and sum(exp()) along the first dimension
 * All operations are performed on Fortran (column-major) order batched matrices
 *
 * @param stream    CUDA stream
 * @param batch     Batch size
 * @param seq       Sequence length
 * @param head      Number of attention heads
 * @param K         Key matrix [head, seq, batch]
 * @param Q         Query matrix [head, seq, batch]
 * @param mask      Attention mask [seq, seq]
 * @param maxsumexp Output buffer for maximum values and sum of exponentials [2, seq, batch]
 */
template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, T *maxsumexp) noexcept;

} // namespace nntile::kernel::flash_maxsumexp
