/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/maximum.cc
 * Per-element maximum of two Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/maximum.hh"
#include "nntile/starpu/maximum.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise maximum operation
/*! @param[in] src: Input tensor for the prod operation
 * @param[inout] dst: Input and output tensor for the maximum operation
 * */
template<typename T>
void maximum_async(const Tensor<T> &src, const Tensor<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Check shapes of base tiles
    if(src.basetile_shape != dst.basetile_shape)
    {
        throw std::runtime_error("src.basetile_shape != dst.basetile_shape");
    }
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        // Get handle for corresponding tiles of src and dst
        auto src_tile_handle = src.get_tile_handle(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        // MPI rank of the destination tile
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Transfer data
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute only on destination node
        if(mpi_rank == dst_tile_rank)
        {
            auto traits = src.get_tile_traits(i);
            starpu::maximum::submit<T>(traits.nelems, src_tile_handle,
                    dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise maximum operation
/*! @param[in] src: Input tensor for the maximum operation
 * @param[inout] dst: Input and output tensor for the maximum operation
 * */
template<typename T>
void maximum(const Tensor<T> &src, const Tensor<T> &dst)
{
    maximum_async<T>(src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void maximum_async<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void maximum_async<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

// Explicit instantiation
template
void maximum<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void maximum<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

} // namespace nntile::tensor
