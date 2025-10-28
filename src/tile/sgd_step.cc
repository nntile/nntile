/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sgd_step.cc
 * SGD with momentum step for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/sgd_step.hh"
#include "nntile/starpu/sgd_step.hh"

namespace nntile::tile
{

//! Asynchronous version of tile-wise fused SGD with momentum step
/*! * @param[in] momentum: momentum coefficient
 * @param[in] lr: learning rate
 * @param[in] weight_decay: coefficient for l2 regularizer
 * @param[in] nesterov: whether to use Nesterov momentum
 * @param[in] grad: Input buffer stored gradient
 * @param[inout] velocity: Input buffer stored velocity (momentum buffer)
 * @param[inout] p: Input buffers with parameter that are updated in the end
 * */
template<typename T>
void sgd_step_async(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
                     const Tile<T> &grad, const Tile<T> &velocity,
                     const Tile<T> &p)
{
    // Check shapes
    if(grad.shape != p.shape)
    {
        throw std::runtime_error("Shapes of gradient and parameters are not equal");
    }
    if(velocity.shape != p.shape)
    {
        throw std::runtime_error("Shapes of velocity and parameters are not equal");
    }
    // Submit task
    starpu::sgd_step.submit<std::tuple<T>>(p.nelems, momentum, lr, weight_decay, nesterov,
                                 grad, velocity, p);
}

//! Blocking version of tile-wise fused SGD with momentum step
/*! * @param[in] momentum: momentum coefficient
 * @param[in] lr: learning rate
 * @param[in] weight_decay: coefficient for l2 regularizer
 * @param[in] nesterov: whether to use Nesterov momentum
 * @param[in] grad: Input buffer stored gradient
 * @param[inout] velocity: Input buffer stored velocity (momentum buffer)
 * @param[inout] p: Input buffers with parameter that are updated in the end
 * */
template<typename T>
void sgd_step(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
               const Tile<T> &grad, const Tile<T> &velocity,
               const Tile<T> &p)
{
    sgd_step_async<T>(momentum, lr, weight_decay, nesterov, grad, velocity, p);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void sgd_step_async<fp32_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
                     const Tile<fp32_t> &grad, const Tile<fp32_t> &velocity,
                     const Tile<fp32_t> &p);

template
void sgd_step_async<fp32_fast_tf32_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
                     const Tile<fp32_fast_tf32_t> &grad, const Tile<fp32_fast_tf32_t> &velocity,
                     const Tile<fp32_fast_tf32_t> &p);

template
void sgd_step_async<fp32_fast_fp16_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
               const Tile<fp32_fast_fp16_t> &grad, const Tile<fp32_fast_fp16_t> &velocity,
               const Tile<fp32_fast_fp16_t> &p);

template
void sgd_step_async<fp32_fast_bf16_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
               const Tile<fp32_fast_bf16_t> &grad, const Tile<fp32_fast_bf16_t> &velocity,
               const Tile<fp32_fast_bf16_t> &p);

template
void sgd_step_async<fp64_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
                     const Tile<fp64_t> &grad, const Tile<fp64_t> &velocity,
                     const Tile<fp64_t> &p);

template
void sgd_step_async<bf16_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
                     const Tile<bf16_t> &grad, const Tile<bf16_t> &velocity,
                     const Tile<bf16_t> &p);

template
void sgd_step_async<fp16_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
                     const Tile<fp16_t> &grad, const Tile<fp16_t> &velocity,
                     const Tile<fp16_t> &p);

// Explicit instantiation
template
void sgd_step<fp32_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
               const Tile<fp32_t> &grad, const Tile<fp32_t> &velocity,
               const Tile<fp32_t> &p);

template
void sgd_step<fp32_fast_tf32_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
               const Tile<fp32_fast_tf32_t> &grad, const Tile<fp32_fast_tf32_t> &velocity,
               const Tile<fp32_fast_tf32_t> &p);

template
void sgd_step<fp32_fast_fp16_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
               const Tile<fp32_fast_fp16_t> &grad, const Tile<fp32_fast_fp16_t> &velocity,
               const Tile<fp32_fast_fp16_t> &p);

template
void sgd_step<fp32_fast_bf16_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
               const Tile<fp32_fast_bf16_t> &grad, const Tile<fp32_fast_bf16_t> &velocity,
               const Tile<fp32_fast_bf16_t> &p);

template
void sgd_step<fp64_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
               const Tile<fp64_t> &grad, const Tile<fp64_t> &velocity,
               const Tile<fp64_t> &p);

template
void sgd_step<bf16_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
               const Tile<bf16_t> &grad, const Tile<bf16_t> &velocity,
               const Tile<bf16_t> &p);

template
void sgd_step<fp16_t>(Scalar momentum, Scalar lr, Scalar weight_decay, bool nesterov,
               const Tile<fp16_t> &grad, const Tile<fp16_t> &velocity,
               const Tile<fp16_t> &p);

} // namespace nntile::tile
