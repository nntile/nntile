/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/lars_step.cc
 * LARS step for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/lars_step.hh"
#include "nntile/starpu/lars_step.hh"

namespace nntile::tile
{

//! Asynchronous version of tile-wise fused LARS step
/*! * @param[in] lr: learning rate
 * @param[in] trust_ratio: trust ratio for LARS
 * @param[in] grad: Input buffer stored gradient
 * @param[inout] p: Input/output buffer with parameters that are updated
 * @param[in] grad_norm: scalar tile containing gradient norm
 * @param[in] p_norm: scalar tile containing parameter norm
 * */
template<typename T>
void lars_step_async(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
                     const Tile<T> &grad, const Tile<T> &p,
                     const Tile<norm_value_t<T>> &grad_norm, const Tile<norm_value_t<T>> &p_norm)
{
    // Check shapes
    if(grad.shape != p.shape)
    {
        throw std::runtime_error("Shapes of gradient and parameters are not equal");
    }
    // Submit task
    starpu::lars_step.submit<std::tuple<T>>(p.nelems, lr, trust_ratio, weight_decay,
                                 grad, p, grad_norm, p_norm);
}

//! Blocking version of tile-wise fused LARS step
/*! * @param[in] lr: learning rate
 * @param[in] trust_ratio: trust ratio for LARS
 * @param[in] grad: Input buffer stored gradient
 * @param[inout] p: Input/output buffer with parameters that are updated
 * @param[in] grad_norm: scalar tile containing gradient norm
 * @param[in] p_norm: scalar tile containing parameter norm
 * */
template<typename T>
void lars_step(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
               const Tile<T> &grad, const Tile<T> &p,
               const Tile<norm_value_t<T>> &grad_norm, const Tile<norm_value_t<T>> &p_norm)
{
    lars_step_async<T>(lr, trust_ratio, weight_decay, grad, p, grad_norm, p_norm);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void lars_step_async<fp32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
                     const Tile<fp32_t> &grad, const Tile<fp32_t> &p,
                     const Tile<norm_value_t<fp32_t>> &grad_norm, const Tile<norm_value_t<fp32_t>> &p_norm);

template
void lars_step_async<fp32_fast_tf32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
                     const Tile<fp32_fast_tf32_t> &grad, const Tile<fp32_fast_tf32_t> &p,
                     const Tile<norm_value_t<fp32_fast_tf32_t>> &grad_norm, const Tile<norm_value_t<fp32_fast_tf32_t>> &p_norm);

template
void lars_step_async<fp32_fast_fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
               const Tile<fp32_fast_fp16_t> &grad, const Tile<fp32_fast_fp16_t> &p,
               const Tile<norm_value_t<fp32_fast_fp16_t>> &grad_norm, const Tile<norm_value_t<fp32_fast_fp16_t>> &p_norm);

template
void lars_step_async<fp32_fast_bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
               const Tile<fp32_fast_bf16_t> &grad, const Tile<fp32_fast_bf16_t> &p,
               const Tile<norm_value_t<fp32_fast_bf16_t>> &grad_norm, const Tile<norm_value_t<fp32_fast_bf16_t>> &p_norm);

template
void lars_step_async<fp64_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
                     const Tile<fp64_t> &grad, const Tile<fp64_t> &p,
                     const Tile<norm_value_t<fp64_t>> &grad_norm, const Tile<norm_value_t<fp64_t>> &p_norm);

template
void lars_step_async<bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
                     const Tile<bf16_t> &grad, const Tile<bf16_t> &p,
                     const Tile<norm_value_t<bf16_t>> &grad_norm, const Tile<norm_value_t<bf16_t>> &p_norm);

template
void lars_step_async<fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
                     const Tile<fp16_t> &grad, const Tile<fp16_t> &p,
                     const Tile<norm_value_t<fp16_t>> &grad_norm, const Tile<norm_value_t<fp16_t>> &p_norm);

// Explicit instantiation
template
void lars_step<fp32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
               const Tile<fp32_t> &grad, const Tile<fp32_t> &p,
               const Tile<norm_value_t<fp32_t>> &grad_norm, const Tile<norm_value_t<fp32_t>> &p_norm);

template
void lars_step<fp32_fast_tf32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
               const Tile<fp32_fast_tf32_t> &grad, const Tile<fp32_fast_tf32_t> &p,
               const Tile<norm_value_t<fp32_fast_tf32_t>> &grad_norm, const Tile<norm_value_t<fp32_fast_tf32_t>> &p_norm);

template
void lars_step<fp32_fast_fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
               const Tile<fp32_fast_fp16_t> &grad, const Tile<fp32_fast_fp16_t> &p,
               const Tile<norm_value_t<fp32_fast_fp16_t>> &grad_norm, const Tile<norm_value_t<fp32_fast_fp16_t>> &p_norm);

template
void lars_step<fp32_fast_bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
               const Tile<fp32_fast_bf16_t> &grad, const Tile<fp32_fast_bf16_t> &p,
               const Tile<norm_value_t<fp32_fast_bf16_t>> &grad_norm, const Tile<norm_value_t<fp32_fast_bf16_t>> &p_norm);

template
void lars_step<fp64_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
               const Tile<fp64_t> &grad, const Tile<fp64_t> &p,
               const Tile<norm_value_t<fp64_t>> &grad_norm, const Tile<norm_value_t<fp64_t>> &p_norm);

template
void lars_step<bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
                     const Tile<bf16_t> &grad, const Tile<bf16_t> &p,
                     const Tile<norm_value_t<bf16_t>> &grad_norm, const Tile<norm_value_t<bf16_t>> &p_norm);

template
void lars_step<fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
                     const Tile<fp16_t> &grad, const Tile<fp16_t> &p,
                     const Tile<norm_value_t<fp16_t>> &grad_norm, const Tile<norm_value_t<fp16_t>> &p_norm);

} // namespace nntile::tile
