/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/adam_step.cc
 * Adam step for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/adam_step.hh"
#include "nntile/starpu/adam_step.hh"

namespace nntile::tile
{

//! Asynchronous version of tile-wise fused Adam step
/*! * @param[in] num_iters: current iteration number
 * @param[in] beta_1: parameter for moving average of first moments
 * @param[in] beta_2: parameter for moving average of second moments
 * @param[in] eps: small scalar to avoid division by zero
 * @param[in] lr: learning rate
 * @param[in] grad: Input buffer stored gradient
 * @param[in] first_moment: Input buffer stored first moments
 * @param[in] second_moment: Input buffer stored second moments
 * @param[inout] p: Input buffers with parameter that are updated in the end
 * */
template<typename T>
void adam_step_async(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
                     const Tile<T> &grad, const Tile<T> &first_moment, const Tile<T> &second_moment,
                     const Tile<T> &p)
{
    // Check shapes
    if(grad.shape != p.shape)
    {
        throw std::runtime_error("Shapes of gradient and parameters are not equal");
    }
    if(first_moment.shape != p.shape)
    {
        throw std::runtime_error("Shapes of first_moment and parameters are not equal");
    }
    if(second_moment.shape != p.shape)
    {
        throw std::runtime_error("Shapes of first_moment and parameters are not equal");
    }
    // Submit task
    starpu::adam_step::submit<T>(num_iter, p.nelems, beta_1, beta_2, eps, lr, weight_decay,
                                 grad, first_moment, second_moment, p);
}

//! Blocking version of tile-wise fused Adam step
/*! * @param[in] num_iters: current iteration number
 * @param[in] beta_1: parameter for moving average of first moments
 * @param[in] beta_2: parameter for moving average of second moments
 * @param[in] eps: small scalar to avoid division by zero
 * @param[in] lr: learning rate
 * @param[in] grad: Input buffer stored gradient
 * @param[in] first_moment: Input buffer stored first moments
 * @param[in] second_moment: Input buffer stored second moments
 * @param[inout] p: Input buffers with parameter that are updated in the end
 * */
template<typename T>
void adam_step(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
               const Tile<T> &grad, const Tile<T> &first_moment, const Tile<T> &second_moment,
               const Tile<T> &p)
{
    adam_step_async<T>(num_iter, beta_1, beta_2, eps, lr, weight_decay, grad, first_moment, second_moment, p);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void adam_step_async<fp32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
                     const Tile<fp32_t> &grad, const Tile<fp32_t> &first_moment, const Tile<fp32_t> &second_moment,
                     const Tile<fp32_t> &p);

template
void adam_step_async<fp32_fast_tf32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
                     const Tile<fp32_fast_tf32_t> &grad, const Tile<fp32_fast_tf32_t> &first_moment, const Tile<fp32_fast_tf32_t> &second_moment,
                     const Tile<fp32_fast_tf32_t> &p);

template
void adam_step_async<fp64_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
                     const Tile<fp64_t> &grad, const Tile<fp64_t> &first_moment, const Tile<fp64_t> &second_moment,
                     const Tile<fp64_t> &p);

template
void adam_step_async<bf16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
                     const Tile<bf16_t> &grad, const Tile<bf16_t> &first_moment, const Tile<bf16_t> &second_moment,
                     const Tile<bf16_t> &p);

// Explicit instantiation
template
void adam_step<fp32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
               const Tile<fp32_t> &grad, const Tile<fp32_t> &first_moment, const Tile<fp32_t> &second_moment,
               const Tile<fp32_t> &p);

template
void adam_step<fp32_fast_tf32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
               const Tile<fp32_fast_tf32_t> &grad, const Tile<fp32_fast_tf32_t> &first_moment, const Tile<fp32_fast_tf32_t> &second_moment,
               const Tile<fp32_fast_tf32_t> &p);

template
void adam_step<fp64_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
               const Tile<fp64_t> &grad, const Tile<fp64_t> &first_moment, const Tile<fp64_t> &second_moment,
               const Tile<fp64_t> &p);

template
void adam_step<bf16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
                     const Tile<bf16_t> &grad, const Tile<bf16_t> &first_moment, const Tile<bf16_t> &second_moment,
                     const Tile<bf16_t> &p);

} // namespace nntile::tile
