/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/lars_step.cc
 * Lars step for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/lars_step.hh"
#include "nntile/starpu/lars_step.hh"

namespace nntile::tile
{

//! Asynchronous version of tile-wise fused Lars step
/*! * @param[in] num_iters: current iteration number
 * @param[in] num_steps: global number of steps for decaying learning rate
 * @param[in] gamma_0: global learning rate
 * @param[in] momentum: parameter for moving average of first moments
 * @param[in] lars_coefficient: lars coefficient for computing local learning rate
 * @param[in] grad: Input buffer stored gradient
 * @param[in] momentum_buffer: Input buffer stored first moments
 * @param[inout] p: Input buffers with parameter that are updated in the end
 * */
template<typename T>
void lars_step_async(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
                     const Tile<T> &grad, const Tile<T> &momentum_buffer, const Tile<T> &p)
{
    // Check shapes
    if(grad.shape != p.shape)
    {
        throw std::runtime_error("Shapes of gradient and parameters are not equal");
    }
    if(momentum_buffer.shape != p.shape)
    {
        throw std::runtime_error("Shapes of first_moment and parameters are not equal");
    }
    // Submit task
    starpu::lars_step::submit<T>(num_iter, p.nelems, num_steps, gamma_0, momentum, weight_decay, lars_coefficient,
                                 grad, momentum_buffer, p);
}

//! Blocking version of tile-wise fused Lars step
/*! * @param[in] num_iters: current iteration number
 * @param[in] num_steps: global number of steps for decaying learning rate
 * @param[in] gamma_0: global learning rate
 * @param[in] momentum: parameter for moving average of first moments
 * @param[in] lars_coefficient: lars coefficient for computing local learning rate
 * @param[in] grad: Input buffer stored gradient
 * @param[in] momentum_buffer: Input buffer stored first moments
 * @param[inout] p: Input buffers with parameter that are updated in the end
 * */
template<typename T>
void lars_step(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
               const Tile<T> &grad, const Tile<T> &momentum_buffer, const Tile<T> &p)
{
    lars_step_async<T>(num_iter, gamma_0, num_steps, momentum, weight_decay, lars_coefficient, grad, momentum_buffer, p);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void lars_step_async<fp32_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
    const Tile<fp32_t> &grad, const Tile<fp32_t> &momentum_buffer, const Tile<fp32_t> &p);

template
void lars_step_async<fp32_fast_tf32_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
    const Tile<fp32_fast_tf32_t> &grad, const Tile<fp32_fast_tf32_t> &momentum_buffer, const Tile<fp32_fast_tf32_t> &p);

template
void lars_step_async<fp32_fast_fp16_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
    const Tile<fp32_fast_fp16_t> &grad, const Tile<fp32_fast_fp16_t> &momentum_buffer, const Tile<fp32_fast_fp16_t> &p);

template
void lars_step_async<fp32_fast_bf16_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
    const Tile<fp32_fast_bf16_t> &grad, const Tile<fp32_fast_bf16_t> &momentum_buffer, const Tile<fp32_fast_bf16_t> &p);

template
void lars_step_async<fp64_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
    const Tile<fp64_t> &grad, const Tile<fp64_t> &momentum_buffer, const Tile<fp64_t> &p);

template
void lars_step_async<bf16_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
    const Tile<bf16_t> &grad, const Tile<bf16_t> &momentum_buffer, const Tile<bf16_t> &p);

// Explicit instantiation
template
void lars_step<fp32_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
               const Tile<fp32_t> &grad, const Tile<fp32_t> &momentum_buffer, const Tile<fp32_t> &p);

template
void lars_step<fp32_fast_tf32_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
               const Tile<fp32_fast_tf32_t> &grad, const Tile<fp32_fast_tf32_t> &momentum_buffer, const Tile<fp32_fast_tf32_t> &p);

template
void lars_step<fp32_fast_fp16_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
               const Tile<fp32_fast_fp16_t> &grad, const Tile<fp32_fast_fp16_t> &momentum_buffer, const Tile<fp32_fast_fp16_t> &p);

template
void lars_step<fp32_fast_bf16_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
               const Tile<fp32_fast_bf16_t> &grad, const Tile<fp32_fast_bf16_t> &momentum_buffer, const Tile<fp32_fast_bf16_t> &p);

template
void lars_step<fp64_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
               const Tile<fp64_t> &grad, const Tile<fp64_t> &momentum_buffer, const Tile<fp64_t> &p);

template
void lars_step<bf16_t>(Index num_iter, Scalar gamma_0, Index num_steps, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
               const Tile<bf16_t> &grad, const Tile<bf16_t> &momentum_buffer, const Tile<bf16_t> &p);

} // namespace nntile::tile
