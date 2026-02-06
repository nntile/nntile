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
#include "nntile/starpu/sgd_step.hh"
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

    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();

    for(Index i = 0; i < p.grid.nelems; ++i)
    {
        // Get handle for corresponding tiles of src and dst
        auto p_tile_handle = p.get_tile_handle(i);
        auto grad_tile_handle = grad.get_tile_handle(i);
        auto velocity_tile_handle = velocity.get_tile_handle(i);
        // MPI rank of the destination tile
        int p_tile_rank = p_tile_handle.mpi_get_rank();
        int grad_tile_rank = grad_tile_handle.mpi_get_rank();
        int velocity_tile_rank = velocity_tile_handle.mpi_get_rank();
        // Transfer data
        grad_tile_handle.mpi_transfer(p_tile_rank, mpi_rank);
        velocity_tile_handle.mpi_transfer(p_tile_rank, mpi_rank);
        // Execute only on destination node
        if(mpi_rank == p_tile_rank)
        {
            auto traits = p.get_tile_traits(i);
            starpu::sgd_step.submit<std::tuple<T>>(num_iter, traits.nelems, momentum, lr, weight_decay, dampening, nesterov,
                                         grad_tile_handle, velocity_tile_handle, p_tile_handle);
        }
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
