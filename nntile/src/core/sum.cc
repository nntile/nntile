/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/sum.cc
 * Sum all elements of a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/sum.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/sum.hh"
#else
#include "nntile/starpu/sum.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

template<typename T>
void sum_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst)
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
        // Insert task
        #ifdef NNTILE_USE_NNHAUL
        haul::sum.submit<std::tuple<T>>(starpu_worker_hint, src.nelems, alpha, src, beta, dst);
        #else
        starpu::sum.submit<std::tuple<T>>(starpu_worker_hint, src.nelems, alpha, src, beta, dst);
        #endif

    }
}

template<typename T>
void sum(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst)
{
    sum_async<T>(starpu_worker_hint, alpha, src, beta, dst);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void sum_async<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst);

template
void sum_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst);

template
void sum_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst);

template
void sum_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst);

template
void sum_async<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst);

template
void sum_async<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst);

template
void sum_async<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst);

// Explicit instantiation
template
void sum<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst);

template
void sum<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst);

template
void sum<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst);

template
void sum<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst);

template
void sum<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst);

template
void sum<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst);

template
void sum<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst);

} // namespace nntile::core
