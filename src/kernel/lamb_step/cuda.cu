/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/lamb_step/cuda.cu
 * LAMB step with buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/lamb_step/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::lamb_step
{

template<typename T>
static __global__
void cuda_kernel(Index num_iter, Index num_elems, Scalar beta_1,
        Scalar beta_2, Scalar eps,
        Scalar lr, Scalar weight_decay,
        Scalar min_trust, Scalar max_trust,
        Scalar alpha, Scalar beta, const T *grad,
        T *first_moment, T *second_moment, T *p)
{
    using Y = typename T::repr_t;

    // For simplicity, compute norms sequentially with thread 0
    __shared__ Y norms[2]; // norms[0] = p_norm_sq, norms[1] = update_norm_sq

    int i = threadIdx.x;

    if(i == 0)
    {
        norms[0] = Y{0.0}; // p_norm_sq
        norms[1] = Y{0.0}; // update_norm_sq

        // Compute norms sequentially
        for(Index j = 0; j < num_elems; ++j)
        {
            Y p_val = Y{p[j]};
            norms[0] += p_val * p_val;

            Y grad_val = Y{grad[j]};
            if (weight_decay != 0)
            {
                grad_val += weight_decay * p_val;
            }

            Y f_val, s_val;
            if(num_iter == 1)
            {
                f_val = (1 - beta_1) * grad_val;
                s_val = ::sqrt(1 - beta_2) * ::fabs(grad_val);
            }
            else
            {
                f_val = Y{first_moment[j]};
                s_val = Y{second_moment[j]};
                f_val = beta_1 * f_val + (1 - beta_1) * grad_val;
                s_val = ::hypot(::sqrt(beta_2) * s_val, ::sqrt(1 - beta_2) * grad_val);
            }

            Y update_val = alpha * f_val / (s_val * beta + eps);
            norms[1] += update_val * update_val;
        }

        // Compute trust ratio
        Y p_norm = ::sqrt(norms[0]);
        Y update_norm = ::sqrt(norms[1]);
        norms[0] = (update_norm > 0) ? (p_norm / update_norm) : 1.0; // trust_ratio
        norms[0] = ::max(min_trust, ::min(max_trust, norms[0]));
    }
    __syncthreads();

    Y trust_ratio = norms[0];

    // Apply updates in parallel
    if(i < num_elems)
    {
        Y p_val = Y{p[i]}, grad_val = Y{grad[i]};
        if (weight_decay != 0)
        {
            grad_val += weight_decay * p_val;
        }

        Y f_val, s_val;
        if(num_iter == 1)
        {
            f_val = (1 - beta_1) * grad_val;
            first_moment[i] = f_val;
            s_val = ::sqrt(1 - beta_2) * ::fabs(grad_val);
            second_moment[i] = s_val;
        }
        else
        {
            f_val = Y{first_moment[i]};
            s_val = Y{second_moment[i]};
            f_val = beta_1 * f_val + (1 - beta_1) * grad_val;
            first_moment[i] = f_val;
            s_val = ::hypot(::sqrt(beta_2) * s_val, ::sqrt(1 - beta_2) * grad_val);
            second_moment[i] = s_val;
        }

        Y denom = s_val * beta + eps;
        Y update = alpha * f_val / denom;
        p[i] = p_val - trust_ratio * update;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index num_iter, Index num_elems, Scalar beta_1,
        Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
        Scalar min_trust, Scalar max_trust,
        const T *grad_, T *first_moment_, T *second_moment_, T *p_)
    noexcept
//! Fused LAMB step operation of buffers
/*!
* @param[in] stream: CUDA stream
* @param[in] num_iter: current iteration number
* @param[in] num_elems: Number of elements in buffers
* @param[in] beta_1: parameter for moving average of first moments
* @param[in] beta_2: parameter for moving average of second moments
* @param[in] eps: small scalar to avoid division by zero
* @param[in] lr: learning rate
* @param[in] weight_decay: coefficient for l2 regularizer
* @param[in] min_trust: minimum trust ratio
* @param[in] max_trust: maximum trust ratio
* @param[in] grad: Input buffer stored gradient
* @param[inout] first_moment: Input buffer stored first moments
* @param[inout] second_moment: Input buffer stored second moments
* @param[inout] p: Input buffers with parameter that are updated in the end
 * */
{
    // Use single block with threads equal to num_elems (capped at 1024)
    dim3 blocks(1), threads(min(1024, (int)num_elems));
    using Y = typename T::repr_t;
    const Scalar alpha = lr / (1.0 - std::pow(beta_1, num_iter));
    const Scalar beta = 1.0 / std::sqrt(1.0 - std::pow(beta_2, num_iter));
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(num_iter, num_elems,
            beta_1, beta_2, eps, lr, weight_decay, min_trust, max_trust,
            alpha, beta, grad_, first_moment_, second_moment_, p_);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr,
        Scalar weight_decay, Scalar min_trust, Scalar max_trust,
        const fp32_t *grad, fp32_t *first_moment,
        fp32_t *second_moment, fp32_t *p)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr,
        Scalar weight_decay, Scalar min_trust, Scalar max_trust,
        const fp64_t *grad, fp64_t *first_moment,
        fp64_t *second_moment, fp64_t *p)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr,
        Scalar weight_decay, Scalar min_trust, Scalar max_trust,
        const bf16_t *grad, bf16_t *first_moment,
        bf16_t *second_moment, bf16_t *p)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr,
        Scalar weight_decay, Scalar min_trust, Scalar max_trust,
        const fp16_t *grad, fp16_t *first_moment,
        fp16_t *second_moment, fp16_t *p)
    noexcept;

} // namespace nntile::kernel::lamb_step
