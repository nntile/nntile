/*! @copyright (c) 2022-present Skolkovo Institute of Science 
 *                              and Technology (Skoltech), Russia.
 *                 2023-present Artificial Intelligence Research 
 *                              Institute (AIRI), Russia. All rights reserved.
 *
 * NNTile is a software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the StarPU runtime system.
 *
 * @file src/kernel/lion_step/cpu.cc
 * Fused Lion step on buffers on CPU
 *
 * @version 1.1.0
 */

#include "nntile/kernel/lion_step/cpu.hh"
#include <cmath>             // for std::fabs
#include "nntile/kernel/cpu.hh" // hypothetical CPU helper

namespace nntile::kernel::lion_step
{

template<typename T>
void cpu(Index num_iter, Index num_elems,
         Scalar beta_1_, Scalar beta_2_,
         Scalar lambda_, Scalar lr_, Scalar weight_decay_,
         const T *grad,   // gradient buffer
         T *first_moment, // buffer for EMA of gradients (m)
         T *p)            // parameter buffer (theta)
    noexcept
//! Fused Lion step on buffers
/*!
 * Implements the Lion optimizer step in-place:
 *
 *   c_t = beta1 * m_{t-1} + (1 - beta1) * grad
 *   theta_t = theta_{t-1} - lr * [ sign(c_t) + lambda * theta_{t-1} ]
 *   m_t = beta2 * m_{t-1} + (1 - beta2) * grad
 *
 * Optionally includes weight_decay by adding weight_decay * p_val to grad_val.
 *
 * @param[in]  num_iter: Current iteration number (not used directly by Lion, but kept for API symmetry)
 * @param[in]  num_elems: Number of elements in buffers
 * @param[in]  beta_1_, beta_2_: Momentum factors
 * @param[in]  lambda_: Coefficient for parameter penalty in the update
 * @param[in]  lr_: Learning rate
 * @param[in]  weight_decay_: L2 regularization coefficient
 * @param[in]  grad: Input buffer for gradients
 * @param[inout] first_moment: Stores the moving average of gradients (m)
 * @param[inout] p: Model parameters updated in-place
 */
{
    using Y = typename T::repr_t;  // e.g. float or double at runtime

    // Convert scalars to underlying type for faster arithmetic
    const Y beta_1      = static_cast<Y>(beta_1_);
    const Y beta_2      = static_cast<Y>(beta_2_);
    const Y lambda_val  = static_cast<Y>(lambda_);
    const Y lr          = static_cast<Y>(lr_);
    const Y wd          = static_cast<Y>(weight_decay_);

    for (Index i = 0; i < num_elems; ++i)
    {
        // Read current parameter & gradient
        Y p_val    = static_cast<Y>(p[i]);
        Y grad_val = static_cast<Y>(grad[i]);

        // Apply weight decay if needed
        if (wd != Y(0))
        {
            grad_val += wd * p_val;
        }

        // Load current momentum (m_{t-1})
        Y m_val = static_cast<Y>(first_moment[i]);

        // c_t = beta1 * m_{t-1} + (1 - beta1) * grad
        const Y c_val = beta_1 * m_val + (Y(1) - beta_1) * grad_val;

        // sign(c_val): +1 if c_val>0, -1 if c_val<0, 0 if c_val=0
        Y sign_c = Y(0);
        if      (c_val > Y(0)) sign_c =  Y(1);
        else if (c_val < Y(0)) sign_c = -Y(1);

        // theta_t = theta_{t-1} - lr * [ sign(c_t ) + lambda * theta_{t-1} ]
        p_val -= lr * (sign_c + lambda_val * p_val);
        p[i] = static_cast<T>(p_val);

        // m_t = beta2 * m_{t-1} + (1 - beta2) * grad
        m_val = beta_2 * m_val + (Y(1) - beta_2) * grad_val;
        first_moment[i] = static_cast<T>(m_val);
    }
}

//-------------------------------------------
// Explicit template instantiations
//-------------------------------------------
template
void cpu<fp32_t>(Index num_iter, Index num_elems,
                 Scalar beta_1, Scalar beta_2,
                 Scalar lambda_, Scalar lr, Scalar weight_decay,
                 const fp32_t *grad,
                 fp32_t *first_moment,
                 fp32_t *p)
    noexcept;

template
void cpu<fp64_t>(Index num_iter, Index num_elems,
                 Scalar beta_1, Scalar beta_2,
                 Scalar lambda_, Scalar lr, Scalar weight_decay,
                 const fp64_t *grad,
                 fp64_t *first_moment,
                 fp64_t *p)
    noexcept;

template
void cpu<bf16_t>(Index num_iter, Index num_elems,
                 Scalar beta_1, Scalar beta_2,
                 Scalar lambda_, Scalar lr, Scalar weight_decay,
                 const bf16_t *grad,
                 bf16_t *first_moment,
                 bf16_t *p)
    noexcept;

} // namespace nntile::kernel::lion_step
        