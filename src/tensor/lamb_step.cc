/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/lamb_step.cc
 * Fused LAMB step operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/lamb_step.hh"
#include "nntile/starpu/lamb_step.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise fuse LAMB step
template<typename T>
void lamb_step_async(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
                    Scalar min_trust, Scalar max_trust,
                    const Tensor<T> &grad, const Tensor<T> &first_moment, const Tensor<T> &second_moment,
                    const Tensor<T> &p)
{
    if (p.matrix_shape != grad.matrix_shape)
    {
        throw std::runtime_error("Parameter shape is not equal to gradient shape");
    }

    if (p.matrix_shape != first_moment.matrix_shape)
    {
        throw std::runtime_error("Parameter shape is not equal to first_moment shape");
    }

    if (p.matrix_shape != second_moment.matrix_shape)
    {
        throw std::runtime_error("Parameter shape is not equal to second_moment shape");
    }

    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();

    for(Index i = 0; i < p.grid.nelems; ++i)
    {
        // Get handle for corresponding tiles of src and dst
        auto p_tile_handle = p.get_tile_handle(i);
        auto grad_tile_handle = grad.get_tile_handle(i);
        auto first_moment_tile_handle = first_moment.get_tile_handle(i);
        auto second_moment_tile_handle = second_moment.get_tile_handle(i);
        // MPI rank of the destination tile
        int p_tile_rank = p_tile_handle.mpi_get_rank();
        int grad_tile_rank = grad_tile_handle.mpi_get_rank();
        int first_moment_tile_rank = first_moment_tile_handle.mpi_get_rank();
        int second_moment_tile_rank = second_moment_tile_handle.mpi_get_rank();
        // Transfer data
        grad_tile_handle.mpi_transfer(p_tile_rank, mpi_rank);
        first_moment_tile_handle.mpi_transfer(p_tile_rank, mpi_rank);
        second_moment_tile_handle.mpi_transfer(p_tile_rank, mpi_rank);
        // Execute only on destination node
        if(mpi_rank == p_tile_rank)
        {
            auto traits = p.get_tile_traits(i);
            starpu::lamb_step.submit<std::tuple<T>>(num_iter, traits.nelems, beta_1, beta_2, eps, lr, weight_decay,
                                         min_trust, max_trust, grad_tile_handle, first_moment_tile_handle,
                                         second_moment_tile_handle, p_tile_handle);
        }
        // Flush cache for the output tile on every node
        p_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise adam_step operation
template<typename T>
void lamb_step(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
               Scalar min_trust, Scalar max_trust,
               const Tensor<T> &grad, const Tensor<T> &first_moment, const Tensor<T> &second_moment,
               const Tensor<T> &p)
{
    lamb_step_async<T>(num_iter, beta_1, beta_2, eps, lr, weight_decay, min_trust, max_trust,
                       grad, first_moment, second_moment, p);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void lamb_step_async<fp32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp32_t> &grad, const Tensor<fp32_t> &first_moment, const Tensor<fp32_t> &second_moment,
                   const Tensor<fp32_t> &p);

template
void lamb_step_async<fp32_fast_tf32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp32_fast_tf32_t> &grad, const Tensor<fp32_fast_tf32_t> &first_moment, const Tensor<fp32_fast_tf32_t> &second_moment,
                   const Tensor<fp32_fast_tf32_t> &p);

template
void lamb_step_async<fp32_fast_fp16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp32_fast_fp16_t> &grad, const Tensor<fp32_fast_fp16_t> &first_moment, const Tensor<fp32_fast_fp16_t> &second_moment,
                   const Tensor<fp32_fast_fp16_t> &p);

template
void lamb_step_async<fp32_fast_bf16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp32_fast_bf16_t> &grad, const Tensor<fp32_fast_bf16_t> &first_moment, const Tensor<fp32_fast_bf16_t> &second_moment,
                   const Tensor<fp32_fast_bf16_t> &p);

template
void lamb_step_async<fp64_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp64_t> &grad, const Tensor<fp64_t> &first_moment, const Tensor<fp64_t> &second_moment,
                   const Tensor<fp64_t> &p);

template
void lamb_step_async<bf16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<bf16_t> &grad, const Tensor<bf16_t> &first_moment, const Tensor<bf16_t> &second_moment,
                   const Tensor<bf16_t> &p);

template
void lamb_step_async<fp16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp16_t> &grad, const Tensor<fp16_t> &first_moment, const Tensor<fp16_t> &second_moment,
                   const Tensor<fp16_t> &p);

// Explicit instantiation
template
void lamb_step<fp32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp32_t> &grad, const Tensor<fp32_t> &first_moment, const Tensor<fp32_t> &second_moment,
                   const Tensor<fp32_t> &p);

template
void lamb_step<fp32_fast_tf32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp32_fast_tf32_t> &grad, const Tensor<fp32_fast_tf32_t> &first_moment, const Tensor<fp32_fast_tf32_t> &second_moment,
                   const Tensor<fp32_fast_tf32_t> &p);

template
void lamb_step<fp32_fast_fp16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp32_fast_fp16_t> &grad, const Tensor<fp32_fast_fp16_t> &first_moment, const Tensor<fp32_fast_fp16_t> &second_moment,
                   const Tensor<fp32_fast_fp16_t> &p);

template
void lamb_step<fp32_fast_bf16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp32_fast_bf16_t> &grad, const Tensor<fp32_fast_bf16_t> &first_moment, const Tensor<fp32_fast_bf16_t> &second_moment,
                   const Tensor<fp32_fast_bf16_t> &p);

template
void lamb_step<fp64_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp64_t> &grad, const Tensor<fp64_t> &first_moment, const Tensor<fp64_t> &second_moment,
                   const Tensor<fp64_t> &p);

template
void lamb_step<bf16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<bf16_t> &grad, const Tensor<bf16_t> &first_moment, const Tensor<bf16_t> &second_moment,
                   const Tensor<bf16_t> &p);

template
void lamb_step<fp16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    Scalar min_trust, Scalar max_trust,
    const Tensor<fp16_t> &grad, const Tensor<fp16_t> &first_moment, const Tensor<fp16_t> &second_moment,
                   const Tensor<fp16_t> &p);

} // namespace nntile::tensor
