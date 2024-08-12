/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/axpy.cc
 * AXPY for two Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/axpy.hh"
#include "nntile/starpu/axpy.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise axpy operation
/*! @param[in] src: Input tensor for the axpy operation
 * @param[inout] dst: Input and output tensor for the axpy operation
 * */
template<typename T>
void axpy_async(const Tensor<T> &alpha, const Tensor<T> &src,
        const Tensor<T> &dst)
{
    // Check shapes
    if(alpha.shape.size() != 0)
    {
        throw std::runtime_error("alpha.shape.size() != 0");
    }
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Check shapes of base tiles
    if(src.basetile_shape != dst.basetile_shape)
    {
        throw std::runtime_error("src.basetile_shape != dst.basetile_shape");
    }
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    // Scatter alpha
    auto alpha_handle = alpha.get_tile_handle(0);
    int alpha_rank = alpha_handle.mpi_get_rank();
    if(mpi_rank == alpha_rank)
    {
        for(Index i = 0; i < mpi_size; ++i)
        {
            if(i == mpi_rank)
            {
                continue;
            }
            alpha_handle.mpi_transfer(i, mpi_rank);
        }
    }
    else
    {
        alpha_handle.mpi_transfer(mpi_rank, mpi_rank);
    }
    // Launch all the required tasks
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
            starpu::axpy::submit<T>(alpha_handle, traits.nelems,
                    src_tile_handle, dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise axpy operation
/*! @param[in] src: Input tensor for the axpy operation
 * @param[inout] dst: Input and output tensor for the axpy operation
 * */
template<typename T>
void axpy(const Tensor<T> &alpha, const Tensor<T> &src, const Tensor<T> &dst)
{
    axpy_async<T>(alpha, src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void axpy_async<fp32_t>(const Tensor<fp32_t> &alpha, const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void axpy_async<fp64_t>(const Tensor<fp64_t> &alpha, const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

// Explicit instantiation
template
void axpy<fp32_t>(const Tensor<fp32_t> &alpha, const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void axpy<fp64_t>(const Tensor<fp64_t> &alpha, const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

//! Asynchronous tensor-wise axpy operation
/*! @param[in] src: Input tensor for the axpy operation
 * @param[inout] dst: Input and output tensor for the axpy operation
 * */
template<typename T>
void axpy_async(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst)
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
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    // Launch all the required tasks
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
            starpu::axpy::submit<T>(alpha, traits.nelems,
                    src_tile_handle, dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise axpy operation
/*! @param[in] src: Input tensor for the axpy operation
 * @param[inout] dst: Input and output tensor for the axpy operation
 * */
template<typename T>
void axpy(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst)
{
    axpy_async<T>(alpha, src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void axpy_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void axpy_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void axpy_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

// Explicit instantiation
template
void axpy<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void axpy<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void axpy<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

} // namespace nntile::tensor
