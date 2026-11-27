/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/mask_scalar.cc
 * Mask scalar operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/mask_scalar.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/mask_scalar.hh"
#else
#include "nntile/starpu/mask_scalar.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Asynchronous tile-wise mask scalar operation
/*! @param[inout] A: Tile for the element-wise mask scalar operation
 * */
template<typename T>
void mask_scalar_async(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val, const Tile<T> &A,
        Index batch_ndim)
{
    Index effective_batch_ndim = batch_ndim;
    if(mask.ndim != A.ndim-effective_batch_ndim)
    {
        if(batch_ndim == 0 && mask.ndim <= A.ndim)
        {
            effective_batch_ndim = A.ndim - mask.ndim;
        }
        else
        {
            throw std::runtime_error("mask.ndim != A.ndim-batch_ndim");
        }
    }
    for(Index i = 0; i < mask.ndim; ++i)
    {
        if(mask.shape[i] != A.shape[effective_batch_ndim+i])
        {
            throw std::runtime_error("mask.shape[i] != "
                    "A.shape[effective_batch_ndim+i]");
        }
    }
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
    #ifndef NNTILE_USE_NNHAUL
    mask.mpi_transfer(a_rank, mpi_rank);
    #endif
    if(mpi_rank != a_rank)
    {
        return;
    }
    Index nslow, nfast;
    if(effective_batch_ndim == 0)
    {
        nslow = A.matrix_shape[A.ndim][0];
        nfast = A.matrix_shape[A.ndim][1];
    }
    else
    {
        nslow = A.matrix_shape[effective_batch_ndim][1];
        nfast = A.matrix_shape[effective_batch_ndim][0];
    }
    // Submit task without any arguments checked
    #ifdef NNTILE_USE_NNHAUL
    haul::mask_scalar.submit<std::tuple<T>>(starpu_worker_hint,
            nslow, nfast, mask, val, A);
    #else
    starpu::mask_scalar.submit<std::tuple<T>>(starpu_worker_hint,
            nslow, nfast, mask, val, A);
    #endif

}

//! Blocking version of tile-wise mask scalar operation
/*! @param[inout] A: Tile for the element-wise mask scalar operation
 * */
template<typename T>
void mask_scalar(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val, const Tile<T> &A,
        Index batch_ndim)
{
    mask_scalar_async<T>(starpu_worker_hint, mask, val, A, batch_ndim);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void mask_scalar_async<fp32_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_tf32_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_fp16_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_bf16_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp64_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp64_t> &A, Index batch_ndim);

template
void mask_scalar_async<bf16_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<bf16_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp16_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp16_t> &A, Index batch_ndim);

// Explicit instantiation
template
void mask_scalar<fp32_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_t> &A, Index batch_ndim);

template
void mask_scalar<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_tf32_t> &A, Index batch_ndim);

template
void mask_scalar<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_fp16_t> &A, Index batch_ndim);

template
void mask_scalar<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_bf16_t> &A, Index batch_ndim);

template
void mask_scalar<fp64_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp64_t> &A, Index batch_ndim);

template
void mask_scalar<bf16_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<bf16_t> &A, Index batch_ndim);

template
void mask_scalar<fp16_t>(int starpu_worker_hint, const Tile<bool_t> &mask, Scalar val,
        const Tile<fp16_t> &A, Index batch_ndim);

} // namespace nntile::core
