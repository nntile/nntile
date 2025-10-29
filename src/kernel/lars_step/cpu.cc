/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/lars_step/cpu.cc
 * Fused LARS step on buffers on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/lars_step/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::lars_step
{

template<typename T>
void cpu(Index num_elems, Scalar lr, Scalar trust_ratio, Scalar weight_norm,
        Scalar grad_norm, Scalar weight_decay, const T *grad, T *p)
    noexcept
//! Fused LARS step on buffers
/*!
 *
 * @param[in] num_elems: Number of elements in buffers
 * @param[in] lr: learning rate
 * @param[in] trust_ratio: trust ratio for LARS
 * @param[in] weight_norm: pre-computed norm of the weight tensor
 * @param[in] grad_norm: pre-computed norm of the gradient tensor
 * @param[in] weight_decay: coefficient for l2 regularizer
 * @param[in] grad: Input buffer stored gradient
 * @param[inout] p: Input/output buffer with parameters that are updated
 * */
{
    using Y = typename T::repr_t;
    const Y lr_{lr}, trust_ratio_{trust_ratio}, weight_norm_{weight_norm},
          grad_norm_{grad_norm}, weight_decay_{weight_decay};
    // Cycle over buffers
    for(Index i = 0; i < num_elems; ++i)
    {
        // Read values from RAM only once
        Y p_val = static_cast<Y>(p[i]), grad_val = static_cast<Y>(grad[i]);
        if (weight_decay_ != 0)
        {
            grad_val += weight_decay_ * p_val;
        }
        // Compute local learning rate
        Y local_lr = (grad_norm_ > 0) ? lr_ * weight_norm_ / grad_norm_ : lr_;
        // Apply trust ratio clipping
        Y adapted_lr = std::min(local_lr, lr_ * trust_ratio_);
        // Update parameters
        p[i] = static_cast<T>(p_val - adapted_lr * grad_val);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index num_elems, Scalar lr, Scalar trust_ratio, Scalar weight_norm,
        Scalar grad_norm, Scalar weight_decay, const fp32_t *grad, fp32_t *p)
    noexcept;

template
void cpu<fp64_t>(Index num_elems, Scalar lr, Scalar trust_ratio, Scalar weight_norm,
        Scalar grad_norm, Scalar weight_decay, const fp64_t *grad, fp64_t *p)
    noexcept;

template
void cpu<bf16_t>(Index num_elems, Scalar lr, Scalar trust_ratio, Scalar weight_norm,
        Scalar grad_norm, Scalar weight_decay, const bf16_t *grad, bf16_t *p)
    noexcept;

template
void cpu<fp16_t>(Index num_elems, Scalar lr, Scalar trust_ratio, Scalar weight_norm,
        Scalar grad_norm, Scalar weight_decay, const fp16_t *grad, fp16_t *p)
    noexcept;

} // namespace nntile::kernel::lars_step
