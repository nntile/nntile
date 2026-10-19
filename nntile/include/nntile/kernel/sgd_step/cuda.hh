/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sgd_step/cuda.hh
 * Fused SGD with momentum step on CUDA buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::sgd_step
{

template<typename T>
void cuda(cudaStream_t stream, Index num_iter, Index num_elems, Scalar momentum, Scalar lr,
        Scalar weight_decay, Scalar dampening, bool nesterov, const T *grad, T *velocity, T *p)
    noexcept;

//! Fused SGD with momentum step over chunked operands on CUDA
/*! Same operation as cuda() above, but each operand's flat element range is
 * backed by a chain of equal-size chunk allocations (a shorter, exact-size
 * tail on the last chunk). The whole range is processed by ONE kernel with
 * the classic kernel's geometry (one element per thread, 256 threads per
 * block): each block first resolves, once, whether its whole 256-element
 * window lies inside a single chunk of every operand (the common case, as
 * chunk element counts are typically multiples of the block size), and then
 * addresses its elements through plain per-block base pointers -- the
 * per-element arithmetic is byte-for-byte cuda_kernel's, so the result is
 * bitwise the single whole-buffer cuda() call. Blocks whose window
 * straddles a chunk boundary (possible for non-power-of-two chunk element
 * counts) resolve each element through the shared chunk-pointer table.
 *
 * @param[in] stream: CUDA stream
 * @param[in] num_iter: current iteration number
 * @param[in] num_elems: Number of elements in each buffer
 * @param[in] momentum: momentum coefficient
 * @param[in] lr: learning rate
 * @param[in] weight_decay: coefficient for l2 regularizer
 * @param[in] dampening: dampening coefficient for momentum
 * @param[in] nesterov: whether to use Nesterov momentum
 * @param[in] grad_chunks: chunk base pointers of the gradient
 * @param[in] ngrad: number of grad chunks
 * @param[in] vel_chunks: chunk base pointers of the velocity
 * @param[in] nvel: number of velocity chunks
 * @param[in] p_chunks: chunk base pointers of the parameters
 * @param[in] np: number of parameter chunks
 * @param[in] elems_grad: elements in a full grad chunk (uniform)
 * @param[in] elems_vel: elements in a full velocity chunk (uniform)
 * @param[in] elems_p: elements in a full parameter chunk (uniform)
 */
template<typename T>
void chunked(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening,
        bool nesterov, const T *const *grad_chunks, Index ngrad,
        T *const *vel_chunks, Index nvel, T *const *p_chunks, Index np,
        Index elems_grad, Index elems_vel, Index elems_p)
    noexcept;

} // namespace nntile::kernel::sgd_step
