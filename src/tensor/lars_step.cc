/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/lars_step.cc
 * Fuse LARS step operation for Tensor<T>
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/tensor/lars_step.hh"

// Other NNTile headers
#include "nntile/starpu/lars_step.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise fuse LARS step
template<typename T>
void lars_step_async(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
                    const Tensor<T> &grad, const Tensor<T> &p)
{
    if (p.matrix_shape != grad.matrix_shape)
    {
        throw std::runtime_error("Parameter shape is not equal to gradient shape");
    }
    if (p.basetile_shape != grad.basetile_shape)
    {
        throw std::runtime_error("Parameter basetile_shape is not equal to gradient basetile_shape");
    }

    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();

    for(Index i = 0; i < p.grid.nelems; ++i)
    {
        // Get handle for corresponding tiles of src and dst
        auto p_tile_handle = p.get_tile_handle(i);
        auto grad_tile_handle = grad.get_tile_handle(i);
        // MPI rank of the destination tile
        int p_tile_rank = p_tile_handle.mpi_get_rank();
        int grad_tile_rank = grad_tile_handle.mpi_get_rank();
        // Transfer data
        grad_tile_handle.mpi_transfer(p_tile_rank, mpi_rank);
        // Execute only on destination node
        if(mpi_rank == p_tile_rank)
        {
            auto traits = p.get_tile_traits(i);
            starpu::lars_step.submit<std::tuple<T>>(traits.nelems, lr, trust_ratio, weight_norm, grad_norm, weight_decay,
                                         grad_tile_handle, p_tile_handle);
        }
        // Flush cache for the output tile on every node
        p_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise lars_step operation
template<typename T>
void lars_step(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
               const Tensor<T> &grad, const Tensor<T> &p)
{
    lars_step_async<T>(lr, trust_ratio, weight_norm, grad_norm, weight_decay, grad, p);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void lars_step_async<fp32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp32_t> &grad, const Tensor<fp32_t> &p);

template
void lars_step_async<fp32_fast_tf32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp32_fast_tf32_t> &grad, const Tensor<fp32_fast_tf32_t> &p);

template
void lars_step_async<fp32_fast_fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp32_fast_fp16_t> &grad, const Tensor<fp32_fast_fp16_t> &p);

template
void lars_step_async<fp32_fast_bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp32_fast_bf16_t> &grad, const Tensor<fp32_fast_bf16_t> &p);

template
void lars_step_async<fp64_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp64_t> &grad, const Tensor<fp64_t> &p);

template
void lars_step_async<bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<bf16_t> &grad, const Tensor<bf16_t> &p);

template
void lars_step_async<fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp16_t> &grad, const Tensor<fp16_t> &p);

// Explicit instantiation
template
void lars_step<fp32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp32_t> &grad, const Tensor<fp32_t> &p);

template
void lars_step<fp32_fast_tf32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp32_fast_tf32_t> &grad, const Tensor<fp32_fast_tf32_t> &p);

template
void lars_step<fp32_fast_fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp32_fast_fp16_t> &grad, const Tensor<fp32_fast_fp16_t> &p);

template
void lars_step<fp32_fast_bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp32_fast_bf16_t> &grad, const Tensor<fp32_fast_bf16_t> &p);

template
void lars_step<fp64_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp64_t> &grad, const Tensor<fp64_t> &p);

template
void lars_step<bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<bf16_t> &grad, const Tensor<bf16_t> &p);

template
void lars_step<fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_norm, Scalar grad_norm, Scalar weight_decay,
    const Tensor<fp16_t> &grad, const Tensor<fp16_t> &p);

} // namespace nntile::tensor
