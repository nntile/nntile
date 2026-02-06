/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/norm.cc
 * Euclidean norm of all elements in a Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/norm.hh"
#include "nntile/starpu/norm.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Tensor-wise norm
template<typename T>
void norm_async(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst)
{
    // Check dimensions
    if(dst.ndim != 0)
    {
        throw std::runtime_error("dst.ndim != 0");
    }
    if(src.nelems == 0)
    {
        throw std::runtime_error("src.nelems == 0");
    }

    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    constexpr Scalar one = 1.0;
    // go over all tiles
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile_handle = src.get_tile_handle(i);
        int src_tile_rank = src_tile_handle.mpi_get_rank();
        // Get destination tile (only one tile for scalar result)
        auto dst_tile_handle = dst.get_tile_handle(0);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();

        // Transfer data to destination node
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on destination node
        if(mpi_rank == dst_tile_rank)
        {
            auto src_tile_traits = src.get_tile_traits(i);
            Index nelems = src_tile_traits.nelems;
            bool init_first = (i == 0);
            // Insert corresponding task
            if(init_first)
            {
                // The first time we need to initialize with src norm
                starpu::norm.submit<std::tuple<T>>(nelems, alpha,
                    src_tile_handle, beta, dst_tile_handle);
            }
            else
            {
                // The rest of the times we accumulate with beta=1
                starpu::norm.submit<std::tuple<T>>(
                    nelems, alpha, src_tile_handle, one, dst_tile_handle);
            }
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Tensor-wise norm
template<typename T>
void norm(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst)
{
    norm_async<T>(alpha, src, beta, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void norm_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, Scalar beta,
        const Tensor<fp32_t> &dst);

template
void norm_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void norm_async<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src, Scalar beta,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void norm_async<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src, Scalar beta,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void norm_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, Scalar beta,
        const Tensor<fp64_t> &dst);

template
void norm_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst);

template
void norm_async<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src, Scalar beta,
        const Tensor<fp16_t> &dst);

// Explicit instantiation
template
void norm<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, Scalar beta,
        const Tensor<fp32_t> &dst);

template
void norm<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void norm<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src, Scalar beta,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void norm<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src, Scalar beta,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void norm<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, Scalar beta,
        const Tensor<fp64_t> &dst);

template
void norm<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst);

template
void norm<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src, Scalar beta,
        const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
