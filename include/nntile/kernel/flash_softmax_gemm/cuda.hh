/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/flash_softmax_gemm/cuda.hh
 * CUDA kernel to compute softmax((QK')/sqrt(d))V using pre-computed maxsumexp
 *
 * @version 1.1.0
 * */
#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::flash_softmax_gemm
{

//! Compute softmax((QK')/sqrt(d))V using pre-computed maxsumexp on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, const T *maxsumexp,
          const T *V, T *A) noexcept;

} // namespace nntile::kernel::flash_softmax_gemm
