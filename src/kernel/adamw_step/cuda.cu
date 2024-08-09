/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/adamw_step/cuda.cu
 * AdamW step with buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/adamw_step/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::adamw_step
{

template<typename T>
static __global__
void cuda_kernel(Index num_iter, Index num_elems, typename T::repr_t beta_1,
        typename T::repr_t beta_2, typename T::repr_t eps,
        typename T::repr_t lr, typename T::repr_t  weight_decay,
        typename T::repr_t alpha, typename T::repr_t beta, const T *grad,
        T *first_moment, T *second_moment, T *p)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    if(i < num_elems)
    {
        // Read values (param+grad) from RAM only once
        Y p_val = Y{p[i]}, grad_val = Y{grad[i]};
        if (weight_decay != 0)
        {
            p_val *= 1 - lr*weight_decay;
        }
        // Read values (first+second moments) from RAM no more than once and
        // update them in the RAM immediately
        Y f_val, s_val;
        if(num_iter == 1)
        {
            f_val = (1-beta_1) * grad_val;
            first_moment[i] = f_val;
            s_val = ::sqrt(1-beta_2) * ::fabs(grad_val);
            second_moment[i] = s_val;
        }
        else
        {
            f_val = Y{first_moment[i]};
            s_val = Y{second_moment[i]};
            f_val = beta_1*f_val + (1-beta_1)*grad_val;
            first_moment[i] = f_val;
            s_val = ::hypot(::sqrt(beta_2)*s_val, ::sqrt(1-beta_2)*grad_val);
            second_moment[i] = s_val;
        }
        // Update parameters using only data in registers
        Y denom = s_val*beta + eps;
        p[i] = p_val - alpha*f_val/denom;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index num_iter, Index num_elems, Scalar beta_1,
        Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
        const T *grad_, T *first_moment_, T *second_moment_, T *p_)
    noexcept
//! Fused AdamW step operation of buffers
/*!
 *
* @param[in] num_iters: current iteration number
* @param[in] num_elems: Number of elements in buffers
* @param[in] beta_1: parameter for moving average of first moments
* @param[in] beta_2: parameter for moving average of second moments
* @param[in] eps: small scalar to avoid division by zero
* @param[in] lr: learning rate
* @param[in] grad_: Input buffer stored gradient
* @param[in] first_moment_: Input buffer stored first moments
* @param[in] second_moment_: Input buffer stored second moments
* @param[inout] p_: Input buffers with parameter that are updated in the end
 * */
{
    dim3 blocks((num_elems+255)/256), threads(256);
    using Y = typename T::repr_t;
    const Scalar alpha = lr / (1.0 - std::pow(beta_1, num_iter));
    const Scalar beta = 1.0 / std::sqrt(1.0 - std::pow(beta_2, num_iter));
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(num_iter, num_elems,
            Y{beta_1}, Y{beta_2}, Y{eps}, Y{lr}, Y{weight_decay}, Y{alpha},
            Y{beta}, grad_, first_moment_, second_moment_, p_);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr,
        Scalar weight_decay, const fp32_t *grad, fp32_t *first_moment,
        fp32_t *second_moment, fp32_t *p)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr,
        Scalar weight_decay, const fp64_t *grad, fp64_t *first_moment,
        fp64_t *second_moment, fp64_t *p)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr,
        Scalar weight_decay, const bf16_t *grad, bf16_t *first_moment,
        bf16_t *second_moment, bf16_t *p)
    noexcept;

} // namespace nntile::kernel::adamw_step
