/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/flash_softmax_gemm_backward_sumprod_slice/cuda.hh
 * CUDA kernel to compute backward pass of softmax(A)V with fused sumprod and slice
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::flash_softmax_gemm_backward_sumprod_slice
{

//! Compute backward pass of softmax(A)V with fused sumprod and slice on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, const T *maxsumexp,
          const T *dA, const T *V, T *dV, T *sumprod_slice) noexcept;

} // namespace nntile::kernel::flash_softmax_gemm_backward_sumprod_slice
