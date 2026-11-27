/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/relu_inplace.cc
 * Inplace ReLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/relu_inplace.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/relu_inplace.hh"
#else
#include "nntile/starpu/relu_inplace.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Asynchronous tile-wise ReLU operation
/*! @param[inout] A: Tile for the element-wise ReLU operation
 * */
template<typename T>
void relu_inplace_async(int starpu_worker_hint, const Tile<T> &A)
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
        haul::relu_inplace.submit<std::tuple<T>>(starpu_worker_hint, A.nelems, A);
        #else
        starpu::relu_inplace.submit<std::tuple<T>>(starpu_worker_hint, A.nelems, A);
        #endif

    }
}

//! Blocking version of tile-wise ReLU operation
/*! @param[inout] A: Tile for the element-wise ReLU operation
 * */
template<typename T>
void relu_inplace(int starpu_worker_hint, const Tile<T> &A)
{
    relu_inplace_async<T>(starpu_worker_hint, A);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void relu_inplace_async<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &A);

template
void relu_inplace_async<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &A);

template
void relu_inplace_async<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &A);

template
void relu_inplace_async<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &A);

template
void relu_inplace_async<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &A);

template
void relu_inplace_async<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &A);

template
void relu_inplace_async<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &A);

// Explicit instantiation
template
void relu_inplace<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &A);

template
void relu_inplace<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &A);

template
void relu_inplace<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &A);

template
void relu_inplace<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &A);

template
void relu_inplace<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &A);

template
void relu_inplace<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &A);

template
void relu_inplace<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &A);

} // namespace nntile::core
