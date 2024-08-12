/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/clear.cc
 * Clear Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/clear.hh"
#include "nntile/starpu/clear.hh"

namespace nntile::tensor
{

template<typename T>
void clear_async(const Tensor<T> &dst)
{
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        if(mpi_rank == dst_tile_rank)
        {
            starpu::clear::submit(dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void clear(const Tensor<T> &dst)
{
    clear_async<T>(dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void clear_async<fp32_t>(const Tensor<fp32_t> &dst);

template
void clear_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &dst);

template
void clear_async<fp64_t>(const Tensor<fp64_t> &dst);

template
void clear_async<bf16_t>(const Tensor<bf16_t> &dst);

//template
//void clear_async<fp16_t>(const Tensor<fp16_t> &dst);

// Explicit instantiation
template
void clear<fp32_t>(const Tensor<fp32_t> &dst);

template
void clear<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &dst);

template
void clear<fp64_t>(const Tensor<fp64_t> &dst);

template
void clear<bf16_t>(const Tensor<bf16_t> &dst);

//template
//void clear<fp16_t>(const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
