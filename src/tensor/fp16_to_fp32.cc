/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/fp16_to_fp32.cc
 * Convert fp16_t array into fp32_t array
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/fp16_to_fp32.hh"
#include "nntile/starpu/fp16_to_fp32.hh"

namespace nntile::tensor
{

void fp16_to_fp32_async(const Tensor<fp16_t> &src, const Tensor<fp32_t> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    if(src.basetile_shape != dst.basetile_shape)
    {
        throw std::runtime_error("src.basetile_shape != dst.basetile_shape");
    }
    // Launch necessary tasks
    int mpi_rank = starpu_mpi_world_rank();
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
            auto dst_tile_traits = dst.get_tile_traits(i);
            starpu::fp16_to_fp32::submit(dst_tile_traits.nelems,
                    src_tile_handle, dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

void fp16_to_fp32(const Tensor<fp16_t> &src, const Tensor<fp32_t> &dst)
{
    fp16_to_fp32_async(src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

} // namespace nntile::tensor
