/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/scale.cc
 * Scale operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/scale.hh"
#include "nntile/starpu/scale.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

//! Tile-wise scale operation
template<typename T>
void scale_async(Scalar alpha, const Tile<T> &src, const Tile<T> &dst)
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
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        // Insert corresponding task
        starpu::scale.submit<std::tuple<T>>(src.nelems, alpha, src, dst);
    }
}

//! Tile-wise scale operation
template<typename T>
void scale(Scalar alpha, const Tile<T> &src, const Tile<T> &dst)
{
    scale_async<T>(alpha, src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void scale_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void scale_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void scale_async<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void scale_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void scale_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

template
void scale_async<fp16_t>(Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst);

// Explicit instantiation of template
template
void scale<fp32_t>(Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void scale<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void scale<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void scale<fp64_t>(Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void scale<bf16_t>(Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

template
void scale<fp16_t>(Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst);

} // namespace nntile::tile
