/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/add_inplace.cc
 * Add inplace operation for Tile<T>'s
 *
 * @version 1.1.0
 * */

#include "nntile/tile/add_inplace.hh"
#include "nntile/starpu/add_inplace.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

//! Tile-wise add operation
template<typename T>
void add_inplace_async(Scalar alpha, const Tile<T> &src, Scalar beta,
        const Tile<T> &dst)
{
    // Check dimensions
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    // Do nothing if alpha is zero
    if(alpha == 0.0 && beta == 1.)
    {
        return;
    }
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        starpu::add_inplace.submit<std::tuple<T>>(src.nelems, alpha, src, beta,
                dst);
    }
}

//! Tile-wise add operation
template<typename T>
void add_inplace(Scalar alpha, const Tile<T> &src, Scalar beta,
        const Tile<T> &dst)
{
    add_inplace_async<T>(alpha, src, beta, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void add_inplace_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src,
        Scalar beta, const Tile<fp32_t> &dst);

template
void add_inplace_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src,
        Scalar beta, const Tile<bf16_t> &dst);

template
void add_inplace_async<fp16_t>(Scalar alpha, const Tile<fp16_t> &src,
        Scalar beta, const Tile<fp16_t> &dst);

template
void add_inplace_async<fp32_fast_tf32_t>(Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst);

template
void add_inplace_async<fp32_fast_fp16_t>(Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst);

template
void add_inplace_async<fp32_fast_bf16_t>(Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst);

template
void add_inplace_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src,
        Scalar beta, const Tile<fp64_t> &dst);

// Explicit instantiation of template
template
void add_inplace<fp32_t>(Scalar alpha, const Tile<fp32_t> &src,
        Scalar beta, const Tile<fp32_t> &dst);

template
void add_inplace<bf16_t>(Scalar alpha, const Tile<bf16_t> &src,
        Scalar beta, const Tile<bf16_t> &dst);

template
void add_inplace<fp16_t>(Scalar alpha, const Tile<fp16_t> &src,
        Scalar beta, const Tile<fp16_t> &dst);

template
void add_inplace<fp32_fast_tf32_t>(Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst);

template
void add_inplace<fp32_fast_fp16_t>(Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst);

template
void add_inplace<fp32_fast_bf16_t>(Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst);

template
void add_inplace<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst);

} // namespace nntile::tile
