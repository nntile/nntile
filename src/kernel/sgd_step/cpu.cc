/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sgd_step/cpu.cc
 * Fused SGD with momentum step on buffers on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sgd_step/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::sgd_step
{

template<typename T>
void cpu(Index num_elems, Scalar momentum_, Scalar lr_, Scalar weight_decay_,
        bool nesterov, const T *grad, T *velocity, T *p)
    noexcept
//! Fused SGD with momentum step on buffers
/*!
 *
 * @param[in] num_elems: Number of elements in buffers
 * @param[in] momentum_: momentum coefficient
 * @param[in] lr_: learning rate
 * @param[in] weight_decay_: coefficient for l2 regularizer
 * @param[in] nesterov: whether to use Nesterov momentum
 * @param[in] grad: Input buffer stored gradient
 * @param[inout] velocity: Input buffer stored velocity (momentum buffer)
 * @param[inout] p: Input buffers with parameter that are updated in the end
 * */
{
    using Y = typename T::repr_t;
    const Y momentum{momentum_}, lr{lr_}, weight_decay{weight_decay_};
    // Cycle over buffers
    for(Index i = 0; i < num_elems; ++i)
    {
        // Read values (param+grad) from RAM only once
        Y p_val = static_cast<Y>(p[i]), grad_val = static_cast<Y>(grad[i]);
        if (weight_decay != 0)
        {
            grad_val += weight_decay * p_val;
        }
        // Read velocity from RAM
        Y velocity_val = static_cast<Y>(velocity[i]);
        // Update velocity: velocity = momentum * velocity + lr * grad
        velocity_val = momentum * velocity_val + lr * grad_val;
        // Store updated velocity
        velocity[i] = static_cast<T>(velocity_val);
        // Update parameters
        if (nesterov)
        {
            // Nesterov: p = p - lr * (grad + momentum * velocity)
            Y effective_grad = grad_val + momentum * velocity_val;
            p[i] = static_cast<T>(p_val - lr * effective_grad);
        }
        else
        {
            // Standard momentum: p = p - velocity
            p[i] = static_cast<T>(p_val - velocity_val);
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index num_elems, Scalar momentum, Scalar lr,
        Scalar weight_decay, bool nesterov, const fp32_t *grad, fp32_t *velocity, fp32_t *p)
    noexcept;

template
void cpu<fp64_t>(Index num_elems, Scalar momentum, Scalar lr,
        Scalar weight_decay, bool nesterov, const fp64_t *grad, fp64_t *velocity, fp64_t *p)
    noexcept;

template
void cpu<bf16_t>(Index num_elems, Scalar momentum, Scalar lr,
        Scalar weight_decay, bool nesterov, const bf16_t *grad, bf16_t *velocity, bf16_t *p)
    noexcept;

template
void cpu<fp16_t>(Index num_elems, Scalar momentum, Scalar lr,
        Scalar weight_decay, bool nesterov, const fp16_t *grad, fp16_t *velocity, fp16_t *p)
    noexcept;

} // namespace nntile::kernel::sgd_step
