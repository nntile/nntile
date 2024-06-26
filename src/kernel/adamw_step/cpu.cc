/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/adamw_step/cpu.cc
 * Fused AdamW step on buffers on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/adamw_step/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::adamw_step
{

template<typename T>
void cpu(Index num_iter, Index num_elems, T beta_1_, T beta_2_, T eps_,
        T lr_, T weight_decay_, const T *grad_, T *first_moment_,
        T *second_moment_, T *p_)
    noexcept
//! Fused AdamW step on buffers
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
    using Y = typename CPUComputeType<T>::value;
    const Y beta_1{beta_1_}, beta_2{beta_2_}, eps{eps_}, lr{lr_},
          weight_decay{weight_decay_};
    const Y alpha = lr / (1 - std::pow(beta_1, num_iter));
    const Y beta = 1.0 / std::sqrt(1 - std::pow(beta_2, num_iter));
    auto grad = reinterpret_cast<const Y *>(grad_);
    auto first_moment = reinterpret_cast<Y *>(first_moment_);
    auto second_moment = reinterpret_cast<Y *>(second_moment_);
    auto p = reinterpret_cast<Y *>(p_);
    // Cycle over buffers
    for(Index i = 0; i < num_elems; ++i)
    {
        // Read values (param+grad) from RAM only once
        Y p_val=p[i], grad_val=grad[i];
        if (weight_decay != 0)
        {
            p_val *= 1 - lr*weight_decay;
        }
        // Read values (first+second moments) from RAM no more than once and
        // update them in the RAM immediately
        Y f_val, s_val;
        if(num_iter == 1)
        {
            f_val = (Y{1.0}-beta_1) * grad_val;
            first_moment[i] = f_val;
            s_val = std::sqrt(Y{.0}1-beta_2) * std::fabs(grad_val);
            second_moment[i] = s_val;
        }
        else
        {
            f_val = first_moment[i];
            s_val = second_moment[i];
            f_val = beta_1*f_val + (Y{1.0}-beta_1)*grad_val;
            first_moment[i] = f_val;
            s_val = std::hypot(std::sqrt(beta_2)*s_val,
                    std::sqrt(Y{1.0}-beta_2)*grad_val);
            second_moment[i] = s_val;
        }
        // Update parameters using only data in registers
        const Y denom = s_val*beta + eps;
        //T denom = ::sqrt(s_val * beta) + eps;
        p[i] = p_val - alpha*f_val/denom;
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index num_iter, Index num_elems, fp32_t beta_1, fp32_t beta_2,
        fp32_t eps, fp32_t lr, fp32_t weight_decay, const fp32_t* grad,
        fp32_t* first_moment, fp32_t* second_moment, fp32_t* p)
    noexcept;

template
void cpu<fp64_t>(Index num_iter, Index num_elems, fp64_t beta_1, fp64_t beta_2,
        fp64_t eps, fp64_t lr, fp64_t weight_decay, const fp64_t* grad,
        fp64_t* first_moment, fp64_t* second_moment, fp64_t* p)
    noexcept;

} // namespace nntile::kernel::adamw_step
