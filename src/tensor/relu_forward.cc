/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/relu_forward.cc
 * ReLU operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/relu_forward.hh"
#include "nntile/starpu/relu_forward.hh"

namespace nntile::tensor
{

template<typename T>
void relu_forward_async(const Tensor<T> &src, const Tensor<T> &dst)
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
            starpu::relu_forward::submit<T>(dst_tile_traits.nelems,
                    src_tile_handle, dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void relu_forward(const Tensor<T> &src, const Tensor<T> &dst)
{
    relu_forward_async<T>(src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void relu_forward_async<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void relu_forward_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void relu_forward_async<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void relu_forward_async<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

// Explicit instantiation
template
void relu_forward<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void relu_forward<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void relu_forward<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void relu_forward<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

} // namespace nntile::tensor
