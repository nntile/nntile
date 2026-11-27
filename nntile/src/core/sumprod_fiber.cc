/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/sumprod_fiber.cc
 * Sums over fibers into a slice of a product of two Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/sumprod_fiber.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/sumprod_fiber.hh"
#else
#include "nntile/starpu/sumprod_fiber.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

template<typename T>
void sumprod_fiber_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &src1, const Tile<T> &src2,
        Scalar beta, const Tile<T> &dst, Index axis, int redux)
{
#ifdef NNTILE_USE_NNHAUL
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
#endif
    // Check shapes of src1 and src2
    if(src1.shape != src2.shape)
    {
        throw std::runtime_error("src1.shape != src2.shape");
    }
    // Check dimensions
    if(dst.ndim != 1)
    {
        throw std::runtime_error("dst.ndim != 1");
    }
    Index ndim = src1.ndim;
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
    // Check shapes of src1 and dst
    if(src1.shape[axis] != dst.shape[0])
    {
        throw std::runtime_error("src1.shape[axis] != dst.shape[0]");
    }
    // Get sizes
    Index m, n, k;
    m = src1.matrix_shape[axis+1][1];
    n = src1.matrix_shape[axis][0];
    k = src1.shape[axis];
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
    src1.mpi_transfer(dst_rank, mpi_rank);
    #endif
    #ifndef NNTILE_USE_NNHAUL
    src2.mpi_transfer(dst_rank, mpi_rank);
    #endif
    if(mpi_rank == dst_rank)
    {
        #ifdef NNTILE_USE_NNHAUL
        haul::sumprod_fiber.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, alpha, src1, src2,
                beta, dst, 0);
        #else
        starpu::sumprod_fiber.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, alpha, src1, src2,
                beta, dst, 0);
        #endif
  // redux ignored for now
    }
}

//! Tile-wise scalar products along outer axes
template<typename T>
void sumprod_fiber(int starpu_worker_hint, Scalar alpha, const Tile<T> &src1, const Tile<T> &src2, Scalar beta,
        const Tile<T> &dst, Index axis, int redux)
{
#ifdef NNTILE_USE_NNHAUL
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
#endif
    sumprod_fiber_async<T>(starpu_worker_hint, alpha, src1, src2, beta, dst, axis, redux);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void sumprod_fiber_async<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src1,
        const Tile<fp32_t> &src2, Scalar beta, const Tile<fp32_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src1,
        const Tile<fp32_fast_tf32_t> &src2, Scalar beta, const Tile<fp32_fast_tf32_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src1,
        const Tile<fp32_fast_fp16_t> &src2, Scalar beta, const Tile<fp32_fast_fp16_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src1,
        const Tile<fp32_fast_bf16_t> &src2, Scalar beta, const Tile<fp32_fast_bf16_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber_async<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src1,
        const Tile<fp64_t> &src2, Scalar beta, const Tile<fp64_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber_async<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src1,
        const Tile<bf16_t> &src2, Scalar beta, const Tile<bf16_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber_async<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src1,
        const Tile<fp16_t> &src2, Scalar beta, const Tile<fp16_t> &dst,
        Index axis, int redux);

// Explicit instantiation
template
void sumprod_fiber<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src1,
        const Tile<fp32_t> &src2, Scalar beta, const Tile<fp32_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src1,
        const Tile<fp32_fast_tf32_t> &src2, Scalar beta, const Tile<fp32_fast_tf32_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src1,
        const Tile<fp32_fast_fp16_t> &src2, Scalar beta, const Tile<fp32_fast_fp16_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src1,
        const Tile<fp32_fast_bf16_t> &src2, Scalar beta, const Tile<fp32_fast_bf16_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src1,
        const Tile<fp64_t> &src2, Scalar beta, const Tile<fp64_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src1,
        const Tile<bf16_t> &src2, Scalar beta, const Tile<bf16_t> &dst,
        Index axis, int redux);

template
void sumprod_fiber<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src1,
        const Tile<fp16_t> &src2, Scalar beta, const Tile<fp16_t> &dst,
        Index axis, int redux);

} // namespace nntile::core
