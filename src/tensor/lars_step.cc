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
#include "nntile/tile/lars_step.hh"
#include "nntile/starpu/lars_step.hh"
#include "nntile/starpu/config.hh"
#include "nntile/tensor/norm.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise fuse LARS step
template<typename T>
void lars_step_async(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
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

    // Create scalar tensors for norms (always fp32 for consistency with Scalar type)
    TensorTraits norm_traits({}, {});
    std::vector<int> norm_distr(1, 0); // Single tile on rank 0
    Tensor<fp32_t> weight_norm_tensor(norm_traits, norm_distr);
    Tensor<fp32_t> grad_norm_tensor(norm_traits, norm_distr);

    // Compute norms
    constexpr Scalar one = 1.0;
    constexpr Scalar zero = 0.0;
    norm_async<fp32_t>(one, p, zero, weight_norm_tensor);
    norm_async<fp32_t>(one, grad, zero, grad_norm_tensor);

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
            auto p_tile = p.get_tile(i);
            auto grad_tile = grad.get_tile(i);
            auto weight_norm_tile = weight_norm_tensor.get_tile(0);
            auto grad_norm_tile = grad_norm_tensor.get_tile(0);
            tile::lars_step_async<T>(lr, trust_ratio, weight_decay, grad_tile, p_tile, weight_norm_tile, grad_norm_tile);
        }
        // Flush cache for the output tile on every node
        p_tile_handle.mpi_flush();
    }

    // Unregister norm tensors
    weight_norm_tensor.unregister();
    grad_norm_tensor.unregister();
}

//! Blocking version of tensor-wise lars_step operation
template<typename T>
void lars_step(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
               const Tensor<T> &grad, const Tensor<T> &p)
{
    lars_step_async<T>(lr, trust_ratio, weight_decay, grad, p);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void lars_step_async<fp32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp32_t> &grad, const Tensor<fp32_t> &p);

template
void lars_step_async<fp32_fast_tf32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp32_fast_tf32_t> &grad, const Tensor<fp32_fast_tf32_t> &p);

template
void lars_step_async<fp32_fast_fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp32_fast_fp16_t> &grad, const Tensor<fp32_fast_fp16_t> &p);

template
void lars_step_async<fp32_fast_bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp32_fast_bf16_t> &grad, const Tensor<fp32_fast_bf16_t> &p);

template
void lars_step_async<fp64_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp64_t> &grad, const Tensor<fp64_t> &p);

template
void lars_step_async<bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<bf16_t> &grad, const Tensor<bf16_t> &p);

template
void lars_step_async<fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp16_t> &grad, const Tensor<fp16_t> &p);

// Explicit instantiation
template
void lars_step<fp32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp32_t> &grad, const Tensor<fp32_t> &p);

template
void lars_step<fp32_fast_tf32_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp32_fast_tf32_t> &grad, const Tensor<fp32_fast_tf32_t> &p);

template
void lars_step<fp32_fast_fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp32_fast_fp16_t> &grad, const Tensor<fp32_fast_fp16_t> &p);

template
void lars_step<fp32_fast_bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp32_fast_bf16_t> &grad, const Tensor<fp32_fast_bf16_t> &p);

template
void lars_step<fp64_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp64_t> &grad, const Tensor<fp64_t> &p);

template
void lars_step<bf16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<bf16_t> &grad, const Tensor<bf16_t> &p);

template
void lars_step<fp16_t>(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<fp16_t> &grad, const Tensor<fp16_t> &p);

} // namespace nntile::tensor
