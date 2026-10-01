/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/fill.cc
 * Fill operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/fill.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/fill.hh"
#else
#include "nntile/starpu/fill.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Asynchronous tile-wise fill operation
/*! @param[inout] A: Tile for the element-wise fill operation
 * */
template<typename T>
void fill_async(int starpu_worker_hint, Scalar val, const Tile<T> &A)
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
        haul::fill.submit<std::tuple<T>>(starpu_worker_hint, A.nelems, val, A);
        #else
        starpu::fill.submit<std::tuple<T>>(starpu_worker_hint, A.nelems, val, A);
        #endif

    }
}

//! Blocking version of tile-wise flll operation
/*! @param[inout] A: Tile for the element-wise fill operation
 * */
template<typename T>
void fill(int starpu_worker_hint, Scalar val, const Tile<T> &A)
{
    fill_async<T>(starpu_worker_hint, val, A);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void fill_async<fp32_t>(int starpu_worker_hint, Scalar val, const Tile<fp32_t> &A);

template
void fill_async<bf16_t>(int starpu_worker_hint, Scalar val, const Tile<bf16_t> &A);

template
void fill_async<fp16_t>(int starpu_worker_hint, Scalar val, const Tile<fp16_t> &A);

template
void fill_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar val, const Tile<fp32_fast_tf32_t> &A);

template
void fill_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar val, const Tile<fp32_fast_fp16_t> &A);

template
void fill_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar val, const Tile<fp32_fast_bf16_t> &A);

template
void fill_async<fp64_t>(int starpu_worker_hint, Scalar val, const Tile<fp64_t> &A);

// Explicit instantiation
template
void fill<fp32_t>(int starpu_worker_hint, Scalar val, const Tile<fp32_t> &A);

template
void fill<bf16_t>(int starpu_worker_hint, Scalar val, const Tile<bf16_t> &A);

template
void fill<fp16_t>(int starpu_worker_hint, Scalar val, const Tile<fp16_t> &A);

template
void fill<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar val, const Tile<fp32_fast_tf32_t> &A);

template
void fill<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar val, const Tile<fp32_fast_fp16_t> &A);

template
void fill<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar val, const Tile<fp32_fast_bf16_t> &A);

template
void fill<fp64_t>(int starpu_worker_hint, Scalar val, const Tile<fp64_t> &A);

} // namespace nntile::core
