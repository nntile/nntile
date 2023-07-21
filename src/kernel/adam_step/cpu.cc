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
void cpu(Index num_iter, Index num_elems, T beta_1, T beta_2, T eps, T lr,
         const T* grad, T* first_moment, T* second_moment, T* p)
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
 * @param[in] grad: Input buffer stored gradient
 * @param[in] first_moment: Input buffer stored first moments
 * @param[in] second_moment: Input buffer stored second moments
 * @param[inout] p: Input buffers with parameter that are updated in the end
 * */
{
    // Cycle over buffers
    for(Index i = 0; i < num_elems; ++i)
    {
        if (num_iter == 1)
        {
            first_moment[i] = (1 - beta_1) * grad[i];
            second_moment[i] = (1 - beta_2) * grad[i] * grad[i];
        }
        else
        {
            first_moment[i] = beta_1 * first_moment[i] + (1 - beta_1) * grad[i];
            second_moment[i] = beta_2 * second_moment[i] + (1 - beta_2) * grad[i] * grad[i];
        }
        p[i] -= lr / (1 - std::pow(beta_1, num_iter)) * first_moment[i] / (std::sqrt(second_moment[i] / (1 - std::pow(beta_2, num_iter))) + eps);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index num_iter, Index num_elems, fp32_t beta_1, fp32_t beta_2, fp32_t eps, fp32_t lr,
         const fp32_t* grad, fp32_t* first_moment, fp32_t* second_moment, fp32_t* p)
    noexcept;

template
void cpu<fp64_t>(Index num_iter, Index num_elems, fp64_t beta_1, fp64_t beta_2, fp64_t eps, fp64_t lr,
         const fp64_t* grad, fp64_t* first_moment, fp64_t* second_moment, fp64_t* p)
    noexcept;

} // namespace adam_step
} // namespace kernel
} // namespace nntile