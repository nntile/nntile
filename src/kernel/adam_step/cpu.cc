/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/adam_step/cpu.cc
 * Fused Adam step on buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-07-21
 * */

#include "nntile/kernel/adam_step/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace adam_step
{

template<typename T>
void cpu(Index num_iter, Index num_elems, T beta_1, T beta_2, T eps, T lr, T weight_decay,
         T* grad, T* first_moment, T* second_moment, T* p)
    noexcept
//! Fused Adam step on buffers
/*!
 * 
 * @param[in] num_iter: current iteration number
 * @param[in] num_elems: Number of elements in buffers
 * @param[in] beta_1: parameter for moving average of first moments
 * @param[in] beta_2: parameter for moving average of second moments
 * @param[in] eps: small scalar to avoid division by zero
 * @param[in] lr: learning rate
 * @param[in] weight_decay: coefficient for l2 regularizer
 * @param[in] grad: Input buffer stored gradient, can be updated if weight_decay > 0
 * @param[in] first_moment: Input buffer stored first moments
 * @param[in] second_moment: Input buffer stored square root of second moments for stability
 * @param[inout] p: Input buffers with parameter that are updated in the end
 * */
{
    T alpha = lr / (1 - ::pow(beta_1, num_iter));
    T beta = 1. / (1 - ::pow(beta_2, num_iter));
    // T beta = 1. / ::sqrt(1 - ::pow(beta_2, num_iter));
    // Cycle over buffers
    for(Index i = 0; i < num_elems; ++i)
    {
        // Read values (param+grad) from RAM only once
        T p_val = p[i], grad_val = grad[i];
        if (weight_decay != 0)
        {
            grad_val += weight_decay * p_val;
            grad[i] = grad_val;
        }
        // Read values (first+second moments) from RAM no more than once and
        // update them in the RAM immediately
        T f_val, s_val;
        if(num_iter == 1)
        {
            f_val = (1. - beta_1) * grad_val;
            first_moment[i] = f_val;
            // s_val = ::sqrt(1-beta_2) * ::abs(grad_val);
            s_val = (1. - beta_2) * grad_val * grad_val;
            second_moment[i] = s_val;
        }
        else
        {
            f_val = first_moment[i];
            s_val = second_moment[i];
            f_val = beta_1 * f_val + (1-beta_1) * grad_val;
            first_moment[i] = f_val;
            // s_val = ::hypot(::sqrt(beta_2)*s_val, ::sqrt(1-beta_2)*grad_val);
            s_val = beta_2 * s_val + (1 - beta_2) * grad_val * grad_val;
            second_moment[i] = s_val;
        }
        // Update parameters using only data in registers
        // T denom = s_val * beta + eps;
        T denom = ::sqrt(s_val * beta) + eps;
        p[i] = p_val - alpha * f_val / denom;
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index num_iter, Index num_elems, fp32_t beta_1, fp32_t beta_2, fp32_t eps,
         fp32_t lr, fp32_t weight_decay,
         fp32_t* grad, fp32_t* first_moment, fp32_t* second_moment, fp32_t* p)
    noexcept;

template
void cpu<fp64_t>(Index num_iter, Index num_elems, fp64_t beta_1, fp64_t beta_2, fp64_t eps,
         fp64_t lr, fp64_t weight_decay,
         fp64_t* grad, fp64_t* first_moment, fp64_t* second_moment, fp64_t* p)
    noexcept;

} // namespace adam_step
} // namespace kernel
} // namespace nntile
