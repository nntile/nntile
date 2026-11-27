/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/sgd_step.cc
 * SGD with momentum step for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/sgd_step.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/sgd_step.hh"
#else
#include "nntile/starpu/sgd_step.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Asynchronous version of tile-wise fused SGD with momentum step
/*! * @param[in] momentum: momentum coefficient
 * @param[in] lr: learning rate
 * @param[in] weight_decay: coefficient for l2 regularizer
 * @param[in] dampening: dampening coefficient for momentum
 * @param[in] nesterov: whether to use Nesterov momentum
 * @param[in] grad: Input buffer stored gradient
 * @param[inout] velocity: Input buffer stored velocity (momentum buffer)
 * @param[inout] p: Input buffers with parameter that are updated in the end
 * */
template<typename T>
void sgd_step_async(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
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
    #ifdef NNTILE_USE_NNHAUL
    int mpi_rank = 0;
    #else
    int mpi_rank = starpu_mpi_world_rank();
    #endif
    #ifdef NNTILE_USE_NNHAUL
    int p_rank = 0;
    #else
    int p_rank = p.mpi_get_rank();
    #endif
    #ifndef NNTILE_USE_NNHAUL
    grad.mpi_transfer(p_rank, mpi_rank);
    #endif
    #ifndef NNTILE_USE_NNHAUL
    velocity.mpi_transfer(p_rank, mpi_rank);
    #endif
    if(mpi_rank == p_rank)
    {
        // Submit task
        #ifdef NNTILE_USE_NNHAUL
        haul::sgd_step.submit<std::tuple<T>>(starpu_worker_hint, num_iter, p.nelems, momentum,
                lr, weight_decay, dampening, nesterov, grad, velocity, p);
        #else
        starpu::sgd_step.submit<std::tuple<T>>(starpu_worker_hint, num_iter, p.nelems, momentum,
                lr, weight_decay, dampening, nesterov, grad, velocity, p);
        #endif

    }
}

//! Blocking version of tile-wise fused SGD with momentum step
/*! * @param[in] momentum: momentum coefficient
 * @param[in] lr: learning rate
 * @param[in] weight_decay: coefficient for l2 regularizer
 * @param[in] dampening: dampening coefficient for momentum
 * @param[in] nesterov: whether to use Nesterov momentum
 * @param[in] grad: Input buffer stored gradient
 * @param[inout] velocity: Input buffer stored velocity (momentum buffer)
 * @param[inout] p: Input buffers with parameter that are updated in the end
 * */
template<typename T>
void sgd_step(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tile<T> &grad, const Tile<T> &velocity,
               const Tile<T> &p)
{
    sgd_step_async<T>(starpu_worker_hint, num_iter, momentum, lr, weight_decay, dampening, nesterov, grad, velocity, p);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void sgd_step_async<fp32_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
                     const Tile<fp32_t> &grad, const Tile<fp32_t> &velocity,
                     const Tile<fp32_t> &p);

template
void sgd_step_async<fp32_fast_tf32_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
                     const Tile<fp32_fast_tf32_t> &grad, const Tile<fp32_fast_tf32_t> &velocity,
                     const Tile<fp32_fast_tf32_t> &p);

template
void sgd_step_async<fp32_fast_fp16_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tile<fp32_fast_fp16_t> &grad, const Tile<fp32_fast_fp16_t> &velocity,
               const Tile<fp32_fast_fp16_t> &p);

template
void sgd_step_async<fp32_fast_bf16_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tile<fp32_fast_bf16_t> &grad, const Tile<fp32_fast_bf16_t> &velocity,
               const Tile<fp32_fast_bf16_t> &p);

template
void sgd_step_async<fp64_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
                     const Tile<fp64_t> &grad, const Tile<fp64_t> &velocity,
                     const Tile<fp64_t> &p);

template
void sgd_step_async<bf16_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
                     const Tile<bf16_t> &grad, const Tile<bf16_t> &velocity,
                     const Tile<bf16_t> &p);

template
void sgd_step_async<fp16_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
                     const Tile<fp16_t> &grad, const Tile<fp16_t> &velocity,
                     const Tile<fp16_t> &p);

// Explicit instantiation
template
void sgd_step<fp32_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tile<fp32_t> &grad, const Tile<fp32_t> &velocity,
               const Tile<fp32_t> &p);

template
void sgd_step<fp32_fast_tf32_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tile<fp32_fast_tf32_t> &grad, const Tile<fp32_fast_tf32_t> &velocity,
               const Tile<fp32_fast_tf32_t> &p);

template
void sgd_step<fp32_fast_fp16_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tile<fp32_fast_fp16_t> &grad, const Tile<fp32_fast_fp16_t> &velocity,
               const Tile<fp32_fast_fp16_t> &p);

template
void sgd_step<fp32_fast_bf16_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tile<fp32_fast_bf16_t> &grad, const Tile<fp32_fast_bf16_t> &velocity,
               const Tile<fp32_fast_bf16_t> &p);

template
void sgd_step<fp64_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tile<fp64_t> &grad, const Tile<fp64_t> &velocity,
               const Tile<fp64_t> &p);

template
void sgd_step<bf16_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tile<bf16_t> &grad, const Tile<bf16_t> &velocity,
               const Tile<bf16_t> &p);

template
void sgd_step<fp16_t>(int starpu_worker_hint, Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tile<fp16_t> &grad, const Tile<fp16_t> &velocity,
               const Tile<fp16_t> &p);

} // namespace nntile::core
