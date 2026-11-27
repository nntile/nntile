/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/maxsumexp.cc
 * Max and sum of exponents of Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/maxsumexp.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/maxsumexp.hh"
#else
#include "nntile/starpu/maxsumexp.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Tile-wise max and sum of exponents along single given axis
template<typename T>
void maxsumexp_async(int starpu_worker_hint, const Tile<T> &src,
        const Tile<T> &dst, Index axis, Scalar beta, int redux)
{
#ifdef NNTILE_USE_NNHAUL
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
#endif
    // Check dimensions
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    Index ndim = src.ndim;
    // Treat special case of ndim=0
    if(ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= ndim)
    {
        throw std::runtime_error("axis >= ndim");
    }
    // Check shapes of src and dst (C-order trailing pair dim).
    if(dst.shape[dst.ndim-1] != 2)
    {
        throw std::runtime_error("dst last dim must be 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(src.shape[i] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i]");
        }
    }
    for(Index i = axis+1; i < src.ndim; ++i)
    {
        if(src.shape[i] != dst.shape[i-1])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i-1]");
        }
    }
    // Get sizes
    Index m, n, k;
    m = src.matrix_shape[axis+1][1];
    n = src.matrix_shape[axis][0];
    k = src.shape[axis];
    // Insert task
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
        haul::maxsumexp.submit<std::tuple<T>>(starpu_worker_hint, m, n, k,
                src, dst, beta, redux);
        #else
        starpu::maxsumexp.submit<std::tuple<T>>(starpu_worker_hint, m, n, k,
                src, dst, beta, redux);
        #endif

    }
}

//! Tile-wise max and sum of exponents along single given axis
template<typename T>
void maxsumexp(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst,
        Index axis, Scalar beta, int redux)
{
#ifdef NNTILE_USE_NNHAUL
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
#endif
    maxsumexp_async<T>(starpu_worker_hint, src, dst, axis, beta, redux);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void maxsumexp_async<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index axis, Scalar beta, int redux);

template
void maxsumexp_async<fp32_fast_tf32_t>(int starpu_worker_hint,
        const Tile<fp32_fast_tf32_t> &src, const Tile<fp32_fast_tf32_t> &dst,
        Index axis, Scalar beta, int redux);

template
void maxsumexp_async<fp32_fast_fp16_t>(int starpu_worker_hint,
        const Tile<fp32_fast_fp16_t> &src, const Tile<fp32_fast_fp16_t> &dst,
        Index axis, Scalar beta, int redux);

template
void maxsumexp_async<fp32_fast_bf16_t>(int starpu_worker_hint,
        const Tile<fp32_fast_bf16_t> &src, const Tile<fp32_fast_bf16_t> &dst,
        Index axis, Scalar beta, int redux);

template
void maxsumexp_async<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index axis, Scalar beta, int redux);

template
void maxsumexp_async<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index axis, Scalar beta, int redux);

template
void maxsumexp_async<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index axis, Scalar beta, int redux);

// Explicit instantiation
template
void maxsumexp<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index axis, Scalar beta, int redux);

template
void maxsumexp<fp32_fast_tf32_t>(int starpu_worker_hint,
        const Tile<fp32_fast_tf32_t> &src, const Tile<fp32_fast_tf32_t> &dst,
        Index axis, Scalar beta, int redux);

template
void maxsumexp<fp32_fast_fp16_t>(int starpu_worker_hint,
        const Tile<fp32_fast_fp16_t> &src, const Tile<fp32_fast_fp16_t> &dst,
        Index axis, Scalar beta, int redux);

template
void maxsumexp<fp32_fast_bf16_t>(int starpu_worker_hint,
        const Tile<fp32_fast_bf16_t> &src, const Tile<fp32_fast_bf16_t> &dst,
        Index axis, Scalar beta, int redux);

template
void maxsumexp<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index axis, Scalar beta, int redux);

template
void maxsumexp<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index axis, Scalar beta, int redux);

template
void maxsumexp<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index axis, Scalar beta, int redux);

} // namespace nntile::core
