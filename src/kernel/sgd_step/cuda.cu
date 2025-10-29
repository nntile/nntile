/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sgd_step/cuda.cu
 * SGD with momentum step with buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sgd_step/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::sgd_step
{

template<typename T>
static __global__
void cuda_kernel(Index num_iter, Index num_elems, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
        const T *grad, T *velocity, T *p)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    if(i < num_elems)
    {
        // Read values (param+grad) from RAM only once
        Y p_val = static_cast<Y>(p[i]), grad_val = static_cast<Y>(grad[i]);
        if (weight_decay != 0)
        {
            grad_val += weight_decay * p_val;
        }
        if (momentum != 0.0)
        {
            Y velocity_val;
            if (num_iter == 1)
            {
                velocity_val = grad_val;
            }
            else
            {
                velocity_val = momentum * static_cast<Y>(velocity[i]) +
                    static_cast<Y>(1.0 - dampening) * grad_val;
            }
            velocity[i] = static_cast<T>(velocity_val);
            if (nesterov)
            {
                grad_val += momentum * velocity_val;
            }
            else
            {
                grad_val = velocity_val;
            }
        }
        p[i] = static_cast<T>(p_val - lr * grad_val);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index num_iter, Index num_elems, Scalar momentum, Scalar lr,
        Scalar weight_decay, Scalar dampening, bool nesterov, const T *grad_, T *velocity_, T *p_)
    noexcept
//! Fused SGD with momentum step operation of buffers
/*!
* @param[in] stream: CUDA stream
* @param[in] num_iter: current iteration number
* @param[in] num_elems: Number of elements in buffers
* @param[in] momentum: momentum coefficient
* @param[in] lr: learning rate
* @param[in] weight_decay: coefficient for l2 regularizer
* @param[in] dampening: dampening coefficient for momentum
* @param[in] nesterov: whether to use Nesterov momentum
* @param[in] grad: Input buffer stored gradient
* @param[inout] velocity: Input buffer stored velocity (momentum buffer)
* @param[inout] p: Input buffers with parameter that are updated in the end
 * */
{
    dim3 blocks((num_elems+255)/256), threads(256);
    using Y = typename T::repr_t;
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(num_iter, num_elems,
            Y{momentum}, Y{lr}, Y{weight_decay}, Y{dampening}, nesterov, grad_, velocity_, p_);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index num_iter, Index num_elems, Scalar momentum,
        Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov, const fp32_t *grad, fp32_t *velocity,
        fp32_t *p)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index num_iter, Index num_elems, Scalar momentum,
        Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov, const fp64_t *grad, fp64_t *velocity,
        fp64_t *p)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index num_iter, Index num_elems, Scalar momentum,
        Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov, const bf16_t *grad, bf16_t *velocity,
        bf16_t *p)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index num_iter, Index num_elems, Scalar momentum,
        Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov, const fp16_t *grad, fp16_t *velocity,
        fp16_t *p)
    noexcept;

} // namespace nntile::kernel::sgd_step
