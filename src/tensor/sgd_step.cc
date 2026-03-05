/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/sgd_step.cc
 * Fuse SGD with momentum step operation for Tensor<T>
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/tensor/sgd_step.hh"

// Other NNTile headers
#include "nntile/tile/sgd_step.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise fuse SGD with momentum step
template<typename T>
void sgd_step_async(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
                    const Tensor<T> &grad, const Tensor<T> &velocity,
                    const Tensor<T> &p)
{
    if (p.matrix_shape != grad.matrix_shape)
    {
        throw std::runtime_error("Parameter shape is not equal to gradient shape");
    }

    if (p.matrix_shape != velocity.matrix_shape)
    {
        throw std::runtime_error("Parameter shape is not equal to velocity shape");
    }

    for(Index i = 0; i < p.grid.nelems; ++i)
    {
        auto p_tile_handle = p.get_tile_handle(i);
        auto grad_tile = grad.get_tile(i);
        auto velocity_tile = velocity.get_tile(i);
        auto p_tile = p.get_tile(i);
        tile::sgd_step_async<T>(num_iter, momentum, lr, weight_decay, dampening,
                nesterov, grad_tile, velocity_tile, p_tile);
        // Flush cache for the output tile on every node
        p_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise sgd_step operation
template<typename T>
void sgd_step(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tensor<T> &grad, const Tensor<T> &velocity,
               const Tensor<T> &p)
{
    sgd_step_async<T>(num_iter, momentum, lr, weight_decay, dampening, nesterov, grad, velocity, p);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void sgd_step_async<fp32_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<fp32_t> &grad, const Tensor<fp32_t> &velocity,
                   const Tensor<fp32_t> &p);

template
void sgd_step_async<fp32_fast_tf32_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<fp32_fast_tf32_t> &grad, const Tensor<fp32_fast_tf32_t> &velocity,
                   const Tensor<fp32_fast_tf32_t> &p);

template
void sgd_step_async<fp32_fast_fp16_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tensor<fp32_fast_fp16_t> &grad, const Tensor<fp32_fast_fp16_t> &velocity,
               const Tensor<fp32_fast_fp16_t> &p);

template
void sgd_step_async<fp32_fast_bf16_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tensor<fp32_fast_bf16_t> &grad, const Tensor<fp32_fast_bf16_t> &velocity,
               const Tensor<fp32_fast_bf16_t> &p);

template
void sgd_step_async<fp64_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<fp64_t> &grad, const Tensor<fp64_t> &velocity,
                   const Tensor<fp64_t> &p);

template
void sgd_step_async<bf16_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<bf16_t> &grad, const Tensor<bf16_t> &velocity,
                   const Tensor<bf16_t> &p);

template
void sgd_step_async<fp16_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<fp16_t> &grad, const Tensor<fp16_t> &velocity,
                   const Tensor<fp16_t> &p);

// Explicit instantiation
template
void sgd_step<fp32_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<fp32_t> &grad, const Tensor<fp32_t> &velocity,
                   const Tensor<fp32_t> &p);

template
void sgd_step<fp32_fast_tf32_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<fp32_fast_tf32_t> &grad, const Tensor<fp32_fast_tf32_t> &velocity,
                   const Tensor<fp32_fast_tf32_t> &p);

template
void sgd_step<fp32_fast_fp16_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tensor<fp32_fast_fp16_t> &grad, const Tensor<fp32_fast_fp16_t> &velocity,
               const Tensor<fp32_fast_fp16_t> &p);

template
void sgd_step<fp32_fast_bf16_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
               const Tensor<fp32_fast_bf16_t> &grad, const Tensor<fp32_fast_bf16_t> &velocity,
               const Tensor<fp32_fast_bf16_t> &p);

template
void sgd_step<fp64_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<fp64_t> &grad, const Tensor<fp64_t> &velocity,
                   const Tensor<fp64_t> &p);

template
void sgd_step<bf16_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<bf16_t> &grad, const Tensor<bf16_t> &velocity,
                   const Tensor<bf16_t> &p);

template
void sgd_step<fp16_t>(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<fp16_t> &grad, const Tensor<fp16_t> &velocity,
                   const Tensor<fp16_t> &p);

} // namespace nntile::tensor
