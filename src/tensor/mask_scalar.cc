/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/mask_scalar.cc
 * Mask scalar operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-22
 * */

#include "nntile/tensor/mask_scalar.hh"
#include "nntile/starpu/mask_scalar.hh"

namespace nntile
{
namespace tensor
{

//! Asynchronous tensor-wise mask calar operation
//
// @param[inout] A: Tensor for the element-wise fill operation
template<typename T>
void mask_scalar_async(const Tensor<bool_t> &mask, T val, const Tensor<T> &A)
{
    // Check shapes and tiles
    for (size_t i = 0; i < A.ndim-1; ++i)
    {
        if (A.shape[i] != mask.shape[i])
        {
            throw std::runtime_error("A.shape[i] != mask.shape[i]");
        }
        if (A.basetile_shape[i] != mask.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[i] != mask.basetile_shape[i]");
        }
    }
    // if(A.shape != mask.shape)
    // {   
    //     std::cout << "A shape = ";
    //     for (auto& s : A.shape)
    //         std::cout << s << " ";
    //     std::cout << std::endl;
    //     std::cout << "mask shape = ";
    //     for (auto& s : mask.shape)
    //         std::cout << s << " ";
    //     std::cout << std::endl;
    //     throw std::runtime_error("A.shape != mask.shape");
    // }
    // if(A.basetile_shape != mask.basetile_shape)
    // {
    //     throw std::runtime_error("A.basetile_shape != mask.basetile_shape");
    // }
    int mpi_rank = starpu_mpi_world_rank();

    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        auto A_tile_handle = A.get_tile_handle(i);
        auto mask_tile_handle = mask.get_tile_handle(i);
        int mask_tile_rank = mask_tile_handle.mpi_get_rank();
        int A_tile_rank = A_tile_handle.mpi_get_rank();
        mask_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);
        // Execute only on node-owner
        if(mpi_rank == A_tile_rank)
        {
            auto tile_traits = A.get_tile_traits(i);
            starpu::mask_scalar::submit<T>(tile_traits.nelems, tile_traits.shape[2], mask_tile_handle, val, A_tile_handle);
        }
        // Flush cache for the output tile on every node
        A_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise mask scalar operation
//
// @param[inout] A: Tensor for the element-wise mask scalar operation
template<typename T>
void mask_scalar(const Tensor<bool_t> &mask, T val, const Tensor<T> &A)
{
    mask_scalar_async<T>(mask, val, A);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void mask_scalar_async<fp32_t>(const Tensor<bool_t> &mask, fp32_t val, const Tensor<fp32_t> &A);

template
void mask_scalar_async<fp64_t>(const Tensor<bool_t> &mask, fp64_t val, const Tensor<fp64_t> &A);

// Explicit instantiation
template
void mask_scalar<fp32_t>(const Tensor<bool_t> &mask, fp32_t val, const Tensor<fp32_t> &A);

template
void mask_scalar<fp64_t>(const Tensor<bool_t> &mask, fp64_t val, const Tensor<fp64_t> &A);

} // namespace tensor
} // namespace nntile

