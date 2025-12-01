/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/lars_step/cuda.cu
 * LARS step with buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/lars_step/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::lars_step
{

template<typename T>
static __global__
void cuda_kernel(Index num_elems, Scalar lr, Scalar trust_ratio,
        Scalar grad_norm, Scalar p_norm, Scalar weight_decay, const T *grad, T *p)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    if(i < num_elems)
    {
        // Cast scalar parameters to appropriate type
        Y lr_val{lr}, trust_ratio_val{trust_ratio}, grad_norm_val{grad_norm};
        Y p_norm_val{p_norm}, weight_decay_val{weight_decay};
        // Read values from RAM only once
        Y p_val = Y{p[i]}, grad_val = Y{grad[i]};
        // Apply weight decay to gradients
        if (weight_decay_val != 0)
        {
            grad_val += weight_decay_val * p_val;
        }
        // Compute local learning rate
        Y local_lr = (grad_norm_val > 0) ? lr_val * p_norm_val / grad_norm_val : lr_val;
        // Apply trust ratio clipping
        Y adapted_lr = ::fmin(local_lr, lr_val * trust_ratio_val);
        // Update parameters
        p[i] = p_val - adapted_lr * grad_val;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index num_elems, Scalar lr, Scalar trust_ratio,
        Scalar grad_norm, Scalar p_norm, Scalar weight_decay, const T *grad, T *p)
    noexcept
//! Fused LARS step operation of buffers
/*!
* @param[in] stream: CUDA stream
* @param[in] num_elems: Number of elements in buffers
* @param[in] lr: learning rate
* @param[in] trust_ratio: trust ratio for LARS
* @param[in] grad_norm: pre-computed norm of the gradient tensor
* @param[in] p_norm: pre-computed norm of the parameter tensor
* @param[in] grad: Input buffer stored gradient
* @param[inout] p: Input/output buffer with parameters that are updated
 * */
{
    dim3 blocks((num_elems+255)/256), threads(256);
    using Y = typename T::repr_t;
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(num_elems,
            lr, trust_ratio, grad_norm, p_norm, weight_decay, grad, p);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index num_elems, Scalar lr, Scalar trust_ratio,
        Scalar grad_norm, Scalar p_norm, Scalar weight_decay, const fp32_t *grad, fp32_t *p)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index num_elems, Scalar lr, Scalar trust_ratio,
        Scalar grad_norm, Scalar p_norm, Scalar weight_decay, const fp64_t *grad, fp64_t *p)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index num_elems, Scalar lr, Scalar trust_ratio,
        Scalar grad_norm, Scalar p_norm, Scalar weight_decay, const bf16_t *grad, bf16_t *p)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index num_elems, Scalar lr, Scalar trust_ratio,
        Scalar grad_norm, Scalar p_norm, Scalar weight_decay, const fp16_t *grad, fp16_t *p)
    noexcept;

} // namespace nntile::kernel::lars_step
