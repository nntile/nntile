/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/prod_inplace.cc
 * Per-element product of two Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/prod_inplace.hh"
#include "nntile/starpu/prod_inplace.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise prod operation
/*! @param[in] src: Input tensor for the prod operation
 * @param[inout] dst: Input and output tensor for the prod operation
 * */
template<typename T>
void prod_inplace_async(const Tensor<T> &src, const Tensor<T> &dst)
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
            starpu::prod_inplace::submit<T>(traits.nelems, src_tile_handle,
                    dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise prod operation
/*! @param[in] src: Input tensor for the prod operation
 * @param[inout] dst: Input and output tensor for the prod operation
 * */
template<typename T>
void prod_inplace(const Tensor<T> &src, const Tensor<T> &dst)
{
    prod_inplace_async<T>(src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void prod_inplace_async<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void prod_inplace_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void prod_inplace_async<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void prod_inplace_async<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

// Explicit instantiation
template
void prod_inplace<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void prod_inplace<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void prod_inplace<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void prod_inplace<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

} // namespace nntile::tensor
