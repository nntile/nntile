/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/copy.cc
 * Copy one tensors into another matching tensor
 *
 * @version 1.0.0
 * */

#include "nntile/tensor/copy.hh"
#include "nntile/starpu/copy.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise copy operation
/*! A simple copy from one tensor into another
 *
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination tensor
 * */
template<typename T>
void copy_async(const Tensor<T> &src, const Tensor<T> &dst)
{
    // Check shapes and tiles
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    if(src.basetile_shape != dst.basetile_shape)
    {
        throw std::runtime_error("src.basetile_shape != dst.basetile_shape");
    }
    // Copy tile-by-tile
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile_handle = src.get_tile_handle(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Transfer source tile to dest node
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on destination node
        if(mpi_rank == dst_tile_rank)
        {
            starpu::copy::submit(src_tile_handle, dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise copy operation
/*! A simple copy from one tile into another
 *
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination tensor
 * */
template<typename T>
void copy(const Tensor<T> &src, const Tensor<T> &dst)
{
    copy_async<T>(src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void copy_async<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void copy_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src, const Tensor<fp32_fast_tf32_t> &dst);

template
void copy_async<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

template
void copy_async<Index>(const Tensor<Index> &src, const Tensor<Index> &dst);

// Explicit instantiation
template
void copy<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void copy<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src, const Tensor<fp32_fast_tf32_t> &dst);

template
void copy<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

template
void copy<Index>(const Tensor<Index> &src, const Tensor<Index> &dst);

} // namespace nntile::tensor
