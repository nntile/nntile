/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/adam_step/cpu.cc
 * Fused Adam step on buffers on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/adam_step/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::adam_step
{

template<typename T>
void cpu(Index num_iter, Index num_elems, Scalar beta_1_, Scalar beta_2_,
        Scalar eps_, Scalar lr_, Scalar weight_decay_, const T *grad,
        T *first_moment, T *second_moment, T *p)
    noexcept
//! Fused Adam step on buffers
/*!
 *
 * @param[in] num_iter: current iteration number
 * @param[in] num_elems: Number of elements in buffers
 * @param[in] beta_1_: parameter for moving average of first moments
 * @param[in] beta_2_: parameter for moving average of second moments
 * @param[in] eps_: small scalar to avoid division by zero
 * @param[in] lr_: learning rate
 * @param[in] weight_decay_: coefficient for l2 regularizer
 * @param[in] grad_: Input buffer stored gradient, can be updated if weight_decay > 0
 * @param[inout] first_moment_: Input buffer stored first moments
 * @param[inout] second_moment_: Input buffer stored square root of second moments for stability
 * @param[inout] p_: Input buffers with parameter that are updated in the end
 * */
{
    using Y = typename T::repr_t;
    const Y beta_1{beta_1_}, beta_2{beta_2_}, eps{eps_}, lr{lr_},
          weight_decay{weight_decay_};
    const Y alpha = lr / (Y{1.0} - std::pow(beta_1, num_iter));
    const Y beta = Y{1.0} / std::sqrt(Y{1.0} - std::pow(beta_2, num_iter));
    // Cycle over buffers
    for(Index i = 0; i < num_elems; ++i)
    {
        // Read values (param+grad) from RAM only once
        Y p_val=static_cast<Y>(p[i]), grad_val = static_cast<Y>(grad[i]);
        if (weight_decay != 0)
        {
            grad_val += weight_decay * p_val;
        }
        // Read values (first+second moments) from RAM no more than once and
        // update them in the RAM immediately
        Y f_val, s_val;
        if(num_iter == 1)
        {
            f_val = (1. - beta_1) * grad_val;
            first_moment[i] = static_cast<T>(f_val);
            s_val = std::sqrt(1-beta_2) * std::fabs(grad_val);
            second_moment[i] = static_cast<T>(s_val);
        }
        else
        {
            f_val = static_cast<Y>(first_moment[i]);
            s_val = static_cast<Y>(second_moment[i]);
            f_val = beta_1*f_val + (1-beta_1)*grad_val;
            first_moment[i] = static_cast<T>(f_val);
            s_val = std::hypot(std::sqrt(beta_2)*s_val,
                    std::sqrt(1-beta_2)*grad_val);
            second_moment[i] = static_cast<T>(s_val);
        }
        // Update parameters using only data in registers
        const Y denom = s_val*beta + eps;
        //T denom = ::sqrt(s_val * beta) + eps;
        p[i] = static_cast<T>(p_val - alpha*f_val/denom);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index num_iter, Index num_elems, Scalar beta_1, Scalar beta_2,
        Scalar eps, Scalar lr, Scalar weight_decay, const fp32_t *grad,
        fp32_t *first_moment, fp32_t *second_moment, fp32_t *p)
    noexcept;

template
void cpu<fp64_t>(Index num_iter, Index num_elems, Scalar beta_1, Scalar beta_2,
        Scalar eps, Scalar lr, Scalar weight_decay, const fp64_t *grad,
        fp64_t *first_moment, fp64_t *second_moment, fp64_t *p)
    noexcept;

template
void cpu<bf16_t>(Index num_iter, Index num_elems, Scalar beta_1, Scalar beta_2,
        Scalar eps, Scalar lr, Scalar weight_decay, const bf16_t *grad,
        bf16_t *first_moment, bf16_t *second_moment, bf16_t *p)
    noexcept;

} // namespace nntile::kernel::adam_step
