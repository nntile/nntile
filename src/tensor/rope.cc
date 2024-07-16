/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/rope.cc
 * Tensor wrappers for the Rotary Positional Embedding
 *
 * @version 1.0.0
 * @author Gleb Karpov
 * @date 2024-06-20
 * */

#include "nntile/tensor/rope.hh"
#include "nntile/starpu/rope.hh"
#include <iostream>

namespace nntile::tensor
{

template<typename T>
void rope_async(const Tensor<T> &sin, const Tensor<T> &cos, 
        const Tensor<T> &src, const Tensor<T> &dst, Index axis)
//! Tensor<T> Rotary Positional Embedding
/*! 
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    // Check dimensions
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }

    if(sin.ndim != cos.ndim)
    {
        throw std::runtime_error("sin.ndim != cos.ndim");
    }

    // Apply per-tile rope asynchronously
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    for(Index i = 0; i < sin.grid.nelems; ++i)
    {
        // Index of current source tile
        auto sin_tile_index = sin.grid.linear_to_index(i);

        // Source tile handle
        auto sin_tile_handle = sin.get_tile_handle(i);
        auto cos_tile_handle = cos.get_tile_handle(i);

        // Get destination tile traits
        auto sin_tile_traits = sin.get_tile_traits(i);

        auto num_repetitions = dst.grid.matrix_shape[3][1];
        auto step_stride = dst.grid.stride[3];

        Index m, k;
        m = sin_tile_traits.matrix_shape[1][0];
        k = sin_tile_traits.matrix_shape[1][1];

        // Set fixed indices of current destination tile
        std::vector<Index> dst_tile_index(dst.ndim);
  
        dst_tile_index[0] = sin_tile_index[0];
        dst_tile_index[1] = sin_tile_index[1];
        dst_tile_index[2] = sin_tile_index[2];

        Index dst_tile_offset = dst_tile_index[0];
        for (Index j = 1; j < 3; ++j) {
            dst_tile_offset += dst_tile_index[j] * dst.grid.stride[j];
        }
        for (Index j = 0; j < num_repetitions; ++j) {
            dst_tile_offset += step_stride * j;
            auto dst_tile_traits = dst.get_tile_traits(dst_tile_offset);
            // Get destination tile handle
            auto dst_tile_handle = dst.get_tile_handle(dst_tile_offset);
            // Get src2 tile handle
            auto src_tile_handle = src.get_tile_handle(dst_tile_offset);
            // MPI rank of the destination tile
            int dst_tile_rank = dst_tile_handle.mpi_get_rank();
            // Transfer data
            sin_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            cos_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            // Execute on destination node
            if(mpi_rank == dst_tile_rank)
            {
                Index l;
                l = dst_tile_traits.matrix_shape[1][1];
                // Insert corresponding task
                starpu::rope::submit<T>(m, k, l, sin_tile_handle, cos_tile_handle,
                        src_tile_handle, dst_tile_handle);
            }
            // Flush cache for the output tile on every node
            dst_tile_handle.mpi_flush();
        }

        // // Loop through all necessary destination tiles across BATCH
        // for(Index j = 0; j < dst.grid.shape[axis]; ++j)
        // {
        //     // Set floating axis
        //     dst_tile_index[axis] = j;
        //     // Get linear offset from index
        //     Index dst_tile_offset = dst.grid.index_to_linear(dst_tile_index);
        //     auto dst_tile_traits = dst.get_tile_traits(dst_tile_offset);
        //     // Get destination tile handle
        //     auto dst_tile_handle = dst.get_tile_handle(dst_tile_offset);
        //     // Get src2 tile handle
        //     auto src_tile_handle = src.get_tile_handle(dst_tile_offset);
        //     // MPI rank of the destination tile
        //     int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        //     // Transfer data
        //     sin_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        //     cos_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        //     src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        //     // Execute on destination node
        //     if(mpi_rank == dst_tile_rank)
        //     {
        //         Index l;
        //         l = dst_tile_traits.matrix_shape[1][1];
        //         // Insert corresponding task
        //         starpu::rope::submit<T>(m, k, l, sin_tile_handle, cos_tile_handle,
        //                 src_tile_handle, dst_tile_handle);
        //     }
        //     // Flush cache for the output tile on every node
        //     dst_tile_handle.mpi_flush();
        // }
        
    }
}

template<typename T>
void rope(const Tensor<T> &sin, const Tensor<T> &cos, 
        const Tensor<T> &src, const Tensor<T> &dst, Index axis)
//! 
/*! Blocking version of rope_async<T>.
 *
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    rope_async<T>(sin, cos, src, dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void rope_async<fp32_t>(const Tensor<fp32_t> &sin, const Tensor<fp32_t> &cos, 
        const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst, Index axis);

template
void rope_async<fp64_t>(const Tensor<fp64_t> &sin, const Tensor<fp64_t> &cos, 
        const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst, Index axis);

template
void rope_async<bf16_t>(const Tensor<bf16_t> &sin, const Tensor<bf16_t> &cos,
        const Tensor<bf16_t> &src, const Tensor<bf16_t> &dst, Index axis);

// Explicit instantiation of template
template
void rope<fp32_t>(const Tensor<fp32_t> &sin, const Tensor<fp32_t> &cos, 
        const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst, Index axis);

template
void rope<fp64_t>(const Tensor<fp64_t> &sin, const Tensor<fp64_t> &cos, 
        const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst, Index axis);

template
void rope<bf16_t>(const Tensor<bf16_t> &sin, const Tensor<bf16_t> &cos,
        const Tensor<bf16_t> &src, const Tensor<bf16_t> &dst, Index axis);

} // namespace tensor
