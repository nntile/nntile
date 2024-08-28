/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/addcdiv.cc
 * Addcdiv operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/addcdiv.hh"
#include "nntile/starpu/addcdiv.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise addcdiv operation
template<typename T>
void addcdiv_async(Scalar val, Scalar eps, const Tensor<T> &nom, const Tensor<T> &denom,
                   const Tensor<T> &src)
{
    if (nom.matrix_shape != denom.matrix_shape) {
        throw std::runtime_error("Nominator shape is not equal to denominator shape");
    }

    if (src.matrix_shape != nom.matrix_shape) {
        throw std::runtime_error("Nominator shape is not equal to source shape");
    }

    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();

    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        // Get handle for corresponding tiles of src and dst
        auto src_tile_handle = src.get_tile_handle(i);
        auto nom_tile_handle = nom.get_tile_handle(i);
        auto denom_tile_handle = denom.get_tile_handle(i);
        // MPI rank of the destination tile
        int nom_tile_rank = nom_tile_handle.mpi_get_rank();
        int denom_tile_rank = denom_tile_handle.mpi_get_rank();
        int src_tile_rank = src_tile_handle.mpi_get_rank();
        // Transfer data
        nom_tile_handle.mpi_transfer(src_tile_rank, mpi_rank);
        denom_tile_handle.mpi_transfer(src_tile_rank, mpi_rank);
        // Execute only on destination node
        if(mpi_rank == src_tile_rank)
        {
            auto traits = src.get_tile_traits(i);
            starpu::addcdiv::submit<T>(val, eps, traits.nelems, nom_tile_handle,
                                       denom_tile_handle, src_tile_handle);
        }
        // Flush cache for the output tile on every node
        src_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise addcdiv operation
template<typename T>
void addcdiv(Scalar val, Scalar eps, const Tensor<T> &nom, const Tensor<T> &denom,
                   const Tensor<T> &src)
{
    addcdiv_async<T>(val, eps, nom, denom, src);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void addcdiv_async<fp32_t>(Scalar val, Scalar eps, const Tensor<fp32_t> &nom,
        const Tensor<fp32_t> &denom, const Tensor<fp32_t> &src);

template
void addcdiv_async<fp32_fast_tf32_t>(Scalar val, Scalar eps, const Tensor<fp32_fast_tf32_t> &nom,
        const Tensor<fp32_fast_tf32_t> &denom, const Tensor<fp32_fast_tf32_t> &src);

template
void addcdiv_async<fp64_t>(Scalar val, Scalar eps, const Tensor<fp64_t> &nom,
        const Tensor<fp64_t> &denom, const Tensor<fp64_t> &src);

template
void addcdiv_async<bf16_t>(Scalar val, Scalar eps, const Tensor<bf16_t> &nom,
        const Tensor<bf16_t> &denom, const Tensor<bf16_t> &src);

// Explicit instantiation
template
void addcdiv<fp32_t>(Scalar val, Scalar eps, const Tensor<fp32_t> &nom,
        const Tensor<fp32_t> &denom, const Tensor<fp32_t> &src);

template
void addcdiv<fp32_fast_tf32_t>(Scalar val, Scalar eps, const Tensor<fp32_fast_tf32_t> &nom,
        const Tensor<fp32_fast_tf32_t> &denom, const Tensor<fp32_fast_tf32_t> &src);

template
void addcdiv<fp64_t>(Scalar val, Scalar eps, const Tensor<fp64_t> &nom,
        const Tensor<fp64_t> &denom, const Tensor<fp64_t> &src);

template
void addcdiv<bf16_t>(Scalar val, Scalar eps, const Tensor<bf16_t> &nom,
        const Tensor<bf16_t> &denom, const Tensor<bf16_t> &src);

} // namespace nntile::tensor
