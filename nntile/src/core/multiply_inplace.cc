/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/multiply_inplace.cc
 * Per-element multiplication of two Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/multiply_inplace.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/multiply_inplace.hh"
#else
#include "nntile/starpu/multiply_inplace.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Asynchronous version of tile-wise multiply operation
/*! @param[in] alpha: Scalar multiplier
 * @param[in] src: Input tile for element-wise multiply operation
 * @param[inout] dst: Input and output tile for the multiply operation
 * */
template<typename T>
void multiply_inplace_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst)
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
        // Submit task
        #ifdef NNTILE_USE_NNHAUL
        haul::multiply_inplace.submit<std::tuple<T>>(starpu_worker_hint, src.nelems, alpha, src,
                dst);
        #else
        starpu::multiply_inplace.submit<std::tuple<T>>(starpu_worker_hint, src.nelems, alpha, src,
                dst);
        #endif

    }
}

//! Blocking version of tile-wise multiply operation
/*! @param[in] alpha: Scalar multiplier
 * @param[in] src: Input tile for element-wise multiply operation
 * @param[inout] dst: Input and output tile for the multiply operation
 * */
template<typename T>
void multiply_inplace(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst)
{
    multiply_inplace_async<T>(starpu_worker_hint, alpha, src, dst);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void multiply_inplace_async<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void multiply_inplace_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void multiply_inplace_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void multiply_inplace_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void multiply_inplace_async<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void multiply_inplace_async<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

template
void multiply_inplace_async<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

// Explicit instantiation
template
void multiply_inplace<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void multiply_inplace<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void multiply_inplace<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void multiply_inplace<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void multiply_inplace<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void multiply_inplace<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void multiply_inplace<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

} // namespace nntile::core
