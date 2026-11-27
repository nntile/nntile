/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/isfinite.cc
 * Check NaN or Inf elements for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/isfinite.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/isfinite.hh"
#else
#include "nntile/starpu/isfinite.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Asynchronous tile-wise check NaN or Inf values
/*! @param[inout] A: Tile for the element-wise check of NaN or Inf
 *  @param[inout] flag: indicator of NaN or Inf values
 * */
template<typename T>
void isfinite_async(int starpu_worker_hint, const Tile<T> &A, const Tile<bool_t> &flag)
{
    #ifdef NNTILE_USE_NNHAUL
    int mpi_rank = 0;
    #else
    int mpi_rank = starpu_mpi_world_rank();
    #endif
    #ifdef NNTILE_USE_NNHAUL
    int a_rank = 0;
    #else
    int a_rank = A.mpi_get_rank();
    #endif
    if(mpi_rank == a_rank)
    {
        // Submit task without any arguments checked
        #ifdef NNTILE_USE_NNHAUL
        haul::isfinite.submit<std::tuple<T>>(starpu_worker_hint, A.nelems, A, flag);
        #else
        starpu::isfinite.submit<std::tuple<T>>(starpu_worker_hint, A.nelems, A, flag);
        #endif

    }
}

//! Blocking version of tile-wise check NaN or Inf values
/*! @param[inout] A: Tile for the check NaN or Inf values
 *  @param[inout] flag: indicator of NaN or Inf values
 * */
template<typename T>
void isfinite(int starpu_worker_hint, const Tile<T> &A, const Tile<bool_t> &flag)
{
    isfinite_async<T>(starpu_worker_hint, A, flag);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void isfinite_async<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<fp32_fast_tf32_t>(int starpu_worker_hint, 
        const Tile<fp32_fast_tf32_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<fp32_fast_fp16_t>(int starpu_worker_hint, 
        const Tile<fp32_fast_fp16_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<fp32_fast_bf16_t>(int starpu_worker_hint, 
        const Tile<fp32_fast_bf16_t> &A, const Tile<bool_t> &flag);

// Explicit instantiation
template
void isfinite<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &A, const Tile<bool_t> &flag);

template
void isfinite<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &A, const Tile<bool_t> &flag);

template
void isfinite<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &A, const Tile<bool_t> &flag);

template
void isfinite<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &A, const Tile<bool_t> &flag);

template
void isfinite<fp32_fast_tf32_t>(int starpu_worker_hint, 
        const Tile<fp32_fast_tf32_t> &A, const Tile<bool_t> &flag);

template
void isfinite<fp32_fast_fp16_t>(int starpu_worker_hint, 
        const Tile<fp32_fast_fp16_t> &A, const Tile<bool_t> &flag);

template
void isfinite<fp32_fast_bf16_t>(int starpu_worker_hint, 
        const Tile<fp32_fast_bf16_t> &A, const Tile<bool_t> &flag);

} // namespace nntile::core
