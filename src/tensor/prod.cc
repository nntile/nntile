/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/prod.cc
 * Per-element product of two Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/prod.hh"
#include "nntile/starpu/prod.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise prod operation
/*! @param[in] src: Input tensor for the prod operation
 * @param[inout] dst: Input and output tensor for the prod operation
 * */
template<typename T>
void prod_async(const Tensor<T> &src1, const Tensor<T> &src2,
        const Tensor<T> &dst)
{
    // Check shapes
    if(src1.shape != src2.shape)
    {
        throw std::runtime_error("src1.shape != src2.shape");
    }
    if(src1.shape != dst.shape)
    {
        throw std::runtime_error("src1.shape != dst.shape");
    }
    // Check shapes of base tiles
    if(src1.basetile_shape != src2.basetile_shape)
    {
        throw std::runtime_error("src1.basetile_shape != src2.basetile_shape");
    }
    if(src1.basetile_shape != dst.basetile_shape)
    {
        throw std::runtime_error("src1.basetile_shape != dst.basetile_shape");
    }
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < src1.grid.nelems; ++i)
    {
        // Get handle for corresponding tiles of src and dst
        auto src1_tile_handle = src1.get_tile_handle(i);
        auto src2_tile_handle = src2.get_tile_handle(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        // MPI rank of the destination tile
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Transfer data
        src1_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        src2_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute only on destination node
        if(mpi_rank == dst_tile_rank)
        {
            auto traits = src1.get_tile_traits(i);
            starpu::prod::submit<T>(traits.nelems, src1_tile_handle,
                    src2_tile_handle, dst_tile_handle);
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
void prod(const Tensor<T> &src1, const Tensor<T> &src2, const Tensor<T> &dst)
{
    prod_async<T>(src1, src2, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void prod_async<fp32_t>(const Tensor<fp32_t> &src1, const Tensor<fp32_t> &src2,
        const Tensor<fp32_t> &dst);

template
void prod_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src1,
        const Tensor<fp32_fast_tf32_t> &src2,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void prod_async<fp64_t>(const Tensor<fp64_t> &src1, const Tensor<fp64_t> &src2,
        const Tensor<fp64_t> &dst);

template
void prod_async<bf16_t>(const Tensor<bf16_t> &src1, const Tensor<bf16_t> &src2,
        const Tensor<bf16_t> &dst);

// Explicit instantiation
template
void prod<fp32_t>(const Tensor<fp32_t> &src1, const Tensor<fp32_t> &src2,
        const Tensor<fp32_t> &dst);

template
void prod<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src1,
        const Tensor<fp32_fast_tf32_t> &src2,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void prod<fp64_t>(const Tensor<fp64_t> &src1, const Tensor<fp64_t> &src2,
        const Tensor<fp64_t> &dst);

template
void prod<bf16_t>(const Tensor<bf16_t> &src1, const Tensor<bf16_t> &src2,
        const Tensor<bf16_t> &dst);

} // namespace nntile::tensor
