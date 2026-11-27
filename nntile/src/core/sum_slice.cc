/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/sum_slice.cc
 * Sum over fibers into a slice of a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/sum_slice.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/sum_slice.hh"
#else
#include "nntile/starpu/sum_slice.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Tile-wise sum_slice
template<typename T>
void sum_slice_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta,
        const Tile<T> &dst, Index axis, int redux)
{
#ifdef NNTILE_USE_NNHAUL
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
#endif
    // Check dimensions
    if(src.ndim - 1 != dst.ndim) // before was src.ndim != dst.ndim
    {
        throw std::runtime_error("src.ndim -1 != dst.ndim");
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

    // check if axis consisted, using two pointers
    for(Index i = 0, j = 0; i < src.ndim; i++)
    {
        if (i == axis) {
            continue;
        }
        if(src.shape[i] != dst.shape[j])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[j]");
        }
        j++;
    }
    // Match scale_slice / kernel layout (see core/scale_slice.cc).
    Index m = 1;
    for(Index i = axis + 1; i < src.ndim; ++i)
    {
        m *= src.shape[i];
    }
    Index n = 1;
    for(Index i = 0; i < axis; ++i)
    {
        n *= src.shape[i];
    }
    const Index k = src.shape[axis];
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
        haul::sum_slice.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, alpha, src, beta, dst,
                0);
        #else
        starpu::sum_slice.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, alpha, src, beta, dst,
                0);
        #endif
  // redux ignored for now
    }
}

//! Tile-wise sum_slice
template<typename T>
void sum_slice(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
        Index axis, int redux)
{
#ifdef NNTILE_USE_NNHAUL
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
#endif
    sum_slice_async<T>(starpu_worker_hint, alpha, src, beta, dst, axis, redux);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void sum_slice_async<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        Scalar beta, const Tile<fp32_t> &dst, Index axis, int redux);

template
void sum_slice_async<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis, int redux);

template
void sum_slice_async<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst, Index axis, int redux);

template
void sum_slice_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        Scalar beta, const Tile<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void sum_slice_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst, Index axis, int redux);

template
void sum_slice_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst, Index axis, int redux);

template
void sum_slice_async<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        Scalar beta, const Tile<fp64_t> &dst, Index axis, int redux);

// Explicit instantiation
template
void sum_slice<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst, Index axis, int redux);

template
void sum_slice<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void sum_slice<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst, Index axis, int redux);

template
void sum_slice<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst, Index axis, int redux);

template
void sum_slice<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst, Index axis, int redux);

template
void sum_slice<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis, int redux);

template
void sum_slice<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst, Index axis, int redux);

} // namespace nntile::core
