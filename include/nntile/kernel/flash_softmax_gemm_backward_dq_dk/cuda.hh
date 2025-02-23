/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/flash_softmax_gemm_backward_dq_dk/cuda.hh
 * CUDA kernel to compute gradients dQ and dK of softmax(A)V
 *
 * @version 1.1.0
 * */
#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::flash_softmax_gemm_backward_dq_dk
{

//! Compute gradients dQ and dK of softmax(A)V on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index batch, Index seq, Index head,
          const T *K, const T *Q, const bool_t *mask, const T *maxsumexp,
          const T *dA, const T *V, const T *sumprod_slice,
          T *dQ, T *dK) noexcept;

} // namespace nntile::kernel::flash_softmax_gemm_backward_dq_dk
