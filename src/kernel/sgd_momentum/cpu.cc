/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is a software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the StarPU runtime system.
 *
 * @file src/kernel/sgd_momentum/cpu.cc
 * Fused SGD with momentum step on buffers on CPU
 *
 * @version 1.0.0
 * */

#include "cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::sgd_momentum
{

template<typename T>
void cpu(Index num_iter, Index num_elems, Scalar momentum_, Scalar lr_,
         Scalar weight_decay_, const T *grad, T *velocity, T *p)
    noexcept
//! Fused SGD with Momentum step on buffers
/*!
 * This function performs the parameter update using SGD with momentum.
 * It first updates the velocity using the momentum coefficient and then
 * updates the parameters. Optionally, L2 weight decay is applied to the gradient.
 *
 * @param[in] num_iter: current iteration number
 * @param[in] num_elems: Number of elements in buffers
 * @param[in] momentum_: momentum coefficient (commonly between 0 and 1)
 * @param[in] lr_: learning rate
 * @param[in] weight_decay_: coefficient for L2 regularization
 * @param[in] grad: Input buffer storing the gradient, which can be updated if weight_decay_ > 0
 * @param[inout] velocity: Buffer storing the momentum term
 * @param[inout] p: Buffer containing the parameters to be updated
 * */
{
    using Y = typename T::repr_t;
    const Y momentum{momentum_}, lr{lr_}, weight_decay{weight_decay_};

    // Cycle over buffers
    for(Index i = 0; i < num_elems; ++i)
    {
        // Read current parameter and gradient from memory
        Y p_val = static_cast<Y>(p[i]);
        Y grad_val = static_cast<Y>(grad[i]);

        // Apply weight decay if enabled
        if(weight_decay != 0)
        {
            grad_val += weight_decay * p_val;
        }

        // Read current velocity from memory
        Y v_val;
        if(num_iter == 1)
        {
            // For the first iteration, initialize velocity as the scaled gradient.
            v_val = lr * grad_val;
        }
        else
        {
            // Update velocity: v = momentum * v + lr * grad
            Y v_prev = static_cast<Y>(velocity[i]);
            v_val = momentum * v_prev + lr * grad_val;
        }
        // Update the velocity buffer
        velocity[i] = static_cast<T>(v_val);

        // Update the parameter: p = p - v
        p[i] = static_cast<T>(p_val - v_val);
    }
}

// Explicit instantiation for supported types
template
void cpu<fp32_t>(Index num_iter, Index num_elems, Scalar momentum, Scalar lr,
                 Scalar weight_decay, const fp32_t *grad, fp32_t *velocity, fp32_t *p)
    noexcept;

template
void cpu<fp64_t>(Index num_iter, Index num_elems, Scalar momentum, Scalar lr,
                 Scalar weight_decay, const fp64_t *grad, fp64_t *velocity, fp64_t *p)
    noexcept;

template
void cpu<bf16_t>(Index num_iter, Index num_elems, Scalar momentum, Scalar lr,
                 Scalar weight_decay, const bf16_t *grad, bf16_t *velocity, bf16_t *p)
    noexcept;

} // namespace nntile::kernel::sgd_momentum

