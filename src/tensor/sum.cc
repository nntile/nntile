/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/sum.cc
 * Sum all elements of a Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/sum.hh"
#include "nntile/tile/sum.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Tensor-wise sum
template<typename T>
void sum_async(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst)
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
    constexpr Scalar one = 1.0;
    auto dst_tile_handle = dst.get_tile_handle(0);
    auto dst_tile = dst.get_tile(0);
    // go over all tiles
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile = src.get_tile(i);
        bool init_first = (i == 0);
        if(init_first)
        {
            tile::sum_async<T>(alpha, src_tile, beta, dst_tile);
        }
        else
        {
            tile::sum_async<T>(alpha, src_tile, one, dst_tile);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Tensor-wise sum
template<typename T>
void sum(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst)
{
    sum_async<T>(alpha, src, beta, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void sum_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, Scalar beta,
        const Tensor<fp32_t> &dst);

template
void sum_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void sum_async<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src, Scalar beta,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void sum_async<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src, Scalar beta,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void sum_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, Scalar beta,
        const Tensor<fp64_t> &dst);

template
void sum_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst);

template
void sum_async<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src, Scalar beta,
        const Tensor<fp16_t> &dst);

// Explicit instantiation
template
void sum<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, Scalar beta,
        const Tensor<fp32_t> &dst);

template
void sum<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void sum<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src, Scalar beta,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void sum<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src, Scalar beta,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void sum<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, Scalar beta,
        const Tensor<fp64_t> &dst);

template
void sum<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst);

template
void sum<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src, Scalar beta,
        const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
