/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/normalize.cc
 * Normalize operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/normalize.hh"
#include "nntile/starpu/normalize.hh"

namespace nntile::tensor
{

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void normalize_async(const Tensor<T> &gamma_beta, const Tensor<T> &src,
        const Tensor<T> &dst, Index size, Scalar eps, Index axis)
{
    // Check gamma_beta
    if(gamma_beta.shape.size() != 1)
    {
        throw std::runtime_error("gamma_beta.shape.size() != 1");
    }
    if(gamma_beta.shape[0] != 2)
    {
        throw std::runtime_error("gamma_beta.shape[0] != 2");
    }
    if(gamma_beta.grid.nelems != 1)
    {
        throw std::runtime_error("gamma_beta.grid.nelems != 1");
    }
    // Check inputs
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    // Input shape dimension shall be at least 1
    if(src.ndim == 0)
    {
        throw std::runtime_error("src.ndim == 0");
    }
    // Check number of elements
    if(size <= 0)
    {
        throw std::runtime_error("size <= 0");
    }
    // Check regularization
    if(eps <= 0)
    {
        throw std::runtime_error("eps <= 0");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes
    if(src.shape[0] != 2)
    {
        throw std::runtime_error("src.shape[0] != 2");
    }
    if(src.basetile_shape[0] != 2)
    {
        throw std::runtime_error("src.basetile_shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != src.shape[i+1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i+1]");
        }
        if(dst.basetile_shape[i] != src.basetile_shape[i+1])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src.basetile_shape[i+1]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
        if(dst.basetile_shape[i] != src.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src.basetile_shape[i]");
        }
    }
    // Obtain gamma_beta on every node
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_size = starpu_mpi_world_size();
    auto gamma_beta_handle = gamma_beta.get_tile_handle(0);
    int gamma_beta_rank = gamma_beta_handle.mpi_get_rank();
    int ret;
    // Transfer data to all nodes from source node
    if(mpi_rank == gamma_beta_rank)
    {
        for(int i = 0; i < mpi_size; ++i)
        {
            // Ignore the source node itself
            if(i == mpi_rank)
            {
                continue;
            }
            gamma_beta_handle.mpi_transfer(i, mpi_rank);
        }
    }
    // Receive data on all other nodes
    else
    {
        gamma_beta_handle.mpi_transfer(mpi_rank, mpi_rank);
    }
    // Apply per-tile normalization asynchronously as needed
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        // Index of current source tile
        auto src_tile_index = src.grid.linear_to_index(i);
        // Source tile traits
        auto src_tile_traits = src.get_tile_traits(i);
        // Source tile handle
        auto src_tile_handle = src.get_tile_handle(i);
        // Set fixed indices of current destination tile
        std::vector<Index> dst_tile_index(dst.ndim);
        for(Index j = 0; j < axis; ++j)
        {
            dst_tile_index[j] = src_tile_index[j+1];
        }
        for(Index j = axis+1; j < dst.ndim; ++j)
        {
            dst_tile_index[j] = src_tile_index[j];
        }
        // Loop through all necessary destination tiles
        for(Index j = 0; j < dst.grid.shape[axis]; ++j)
        {
            // Set floating axis
            dst_tile_index[axis] = j;
            // Get linear offset from index
            Index dst_tile_offset = dst.grid.index_to_linear(dst_tile_index);
            // Get destination tile handle
            auto dst_tile_handle = dst.get_tile_handle(dst_tile_offset);
            // MPI rank of the destination tile
            int dst_tile_rank = dst_tile_handle.mpi_get_rank();
            // Transfer data
            src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            // Execute on destination node
            if(mpi_rank == dst_tile_rank)
            {
                // Get destination tile traits
                auto dst_tile_traits = dst.get_tile_traits(dst_tile_offset);
                // Reshape inputs for simplicity: src -> (2,m,n), dst -> (m,k,n)
                // dst is a part of (m,size,n) tensor
                Index m, n, k;
                m = dst_tile_traits.stride[axis];
                n = dst_tile_traits.matrix_shape[axis+1][1];
                k = dst_tile_traits.shape[axis];
                // Insert corresponding task
                starpu::normalize::submit<T>(m, n, k, size, eps,
                        gamma_beta_handle, src_tile_handle, dst_tile_handle);
            }
            // Flush cache for the output tile on every node
            dst_tile_handle.mpi_flush();
        }
    }
}

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void normalize(const Tensor<T> &gamma_beta, const Tensor<T> &src,
        const Tensor<T> &dst, Index size, Scalar eps, Index axis)
{
    normalize_async<T>(gamma_beta, src, dst, size, eps, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void normalize_async<fp32_t>(const Tensor<fp32_t> &gamma_beta,
        const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst, Index size,
        Scalar eps, Index axis);

template
void normalize_async<fp64_t>(const Tensor<fp64_t> &gamma_beta,
        const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst, Index size,
        Scalar eps, Index axis);

// Explicit instantiation
template
void normalize<fp32_t>(const Tensor<fp32_t> &gamma_beta,
        const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst, Index size,
        Scalar eps, Index axis);

template
void normalize<fp64_t>(const Tensor<fp64_t> &gamma_beta,
        const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst, Index size,
        Scalar eps, Index axis);

} // namespace nntile::tensor
