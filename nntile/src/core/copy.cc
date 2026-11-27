/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/copy.cc
 * Copy one tile into another matching tile
 *
 * @version 1.1.0
 * */

#include "nntile/core/copy.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/copy.hh"
#else
#include "nntile/starpu/copy.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Asynchronous version of tile-wise copy operation
/*! A simple copy from one tile into another
 *
 * @param[in] src: Source tile
 * @param[inout] dst: Destination tile
 * */
template<typename T>
void copy_async(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
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
        haul::copy.submit(starpu_worker_hint, src, dst);
        #else
        starpu::copy.submit(starpu_worker_hint, src, dst);
        #endif

    }
}

//! Blocking version of tile-wise copy operation
/*! A simple copy from one tile into another
 *
 * @param[in] src: Source tile
 * @param[inout] dst: Destination tile
 * */
template<typename T>
void copy(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    copy_async<T>(starpu_worker_hint, src, dst);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void copy_async<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void copy_async<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void copy_async<int64_t>(int starpu_worker_hint, const Tile<int64_t> &src, const Tile<int64_t> &dst);

template
void copy_async<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void copy_async<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

template
void copy_async<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void copy_async<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void copy_async<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void copy_async<bool_t>(int starpu_worker_hint, const Tile<bool_t> &src, const Tile<bool_t> &dst);

// Explicit instantiation
template
void copy<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void copy<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void copy<int64_t>(int starpu_worker_hint, const Tile<int64_t> &src, const Tile<int64_t> &dst);

template
void copy<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void copy<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

template
void copy<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void copy<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void copy<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void copy<bool_t>(int starpu_worker_hint, const Tile<bool_t> &src, const Tile<bool_t> &dst);

} // namespace nntile::core
