/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/transpose.cc
 * Transpose operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/transpose.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/transpose.hh"
#else
#include "nntile/starpu/transpose.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Tile-wise transpose operation
template<typename T>
void transpose_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index ndim)
{
    // Check dimensions
    if(ndim <= 0 or ndim >= src.ndim)
    {
        throw std::runtime_error("ndim <= 0 or ndim >= src.ndim");
    }
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(src.shape[(i+ndim) % dst.ndim] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[(i+ndim) % dst.ndim] != "
                    "dst.shape[i]");
        }
    }
    #ifdef NNTILE_USE_NNHAUL
    int mpi_rank = 0;
    #else
    int mpi_rank = starpu_mpi_world_rank();
    #endif
    #ifdef NNTILE_USE_NNHAUL
    int dst_rank = 0;
    #else
    int dst_rank = dst.mpi_get_rank();
    #endif
    #ifndef NNTILE_USE_NNHAUL
    src.mpi_transfer(dst_rank, mpi_rank);
    #endif
    if(mpi_rank == dst_rank)
    {
        #ifdef NNTILE_USE_NNHAUL
        haul::transpose.submit<std::tuple<T>>(starpu_worker_hint,
                src.matrix_shape[ndim][1],
                src.matrix_shape[ndim][0], alpha, src, dst);
        #else
        starpu::transpose.submit<std::tuple<T>>(starpu_worker_hint,
                src.matrix_shape[ndim][1],
                src.matrix_shape[ndim][0], alpha, src, dst);
        #endif

    }
}

//! Tile-wise transpose operation
template<typename T>
void transpose(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index ndim)
{
    transpose_async<T>(starpu_worker_hint, alpha, src, dst, ndim);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation of template
template
void transpose_async<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index ndim);

template
void transpose_async<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index ndim);

template
void transpose_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index ndim);

template
void transpose_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index ndim);

template
void transpose_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index ndim);

template
void transpose_async<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index ndim);

template
void transpose_async<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index ndim);

// Explicit instantiation of template
template
void transpose<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index ndim);

template
void transpose<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index ndim);

template
void transpose<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index ndim);

template
void transpose<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index ndim);

template
void transpose<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index ndim);

template
void transpose<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index ndim);

template
void transpose<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index ndim);

} // namespace nntile::core
