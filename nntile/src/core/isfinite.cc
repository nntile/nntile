/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/isfinite.cc
 * Check NaN or Inf elements for Tile<T>
 *
 * @version 1.1.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include "nntile/core/isfinite.hh"
#include "nntile/kernel_scope.hh"
#include "nntile/nnhaul/ops/isfinite.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Asynchronous tile-wise check NaN or Inf values
/*! @param[inout] A: Tile for the element-wise check of NaN or Inf
 *  @param[inout] flag: indicator of NaN or Inf values
 * */
template<typename T>
void isfinite_async(int worker_hint, const Tile<T> &A, const Tile<bool_t> &flag)
{
    nntile::require_kernel_family("isfinite");
    // Submit task without any arguments checked
    nnhaul::isfinite.submit<std::tuple<T>>(worker_hint, A.nelems, A, flag);
}

//! Blocking version of tile-wise check NaN or Inf values
/*! @param[inout] A: Tile for the check NaN or Inf values
 *  @param[inout] flag: indicator of NaN or Inf values
 * */
template<typename T>
void isfinite(int worker_hint, const Tile<T> &A, const Tile<bool_t> &flag)
{
    nntile::require_kernel_family("isfinite");
    isfinite_async<T>(worker_hint, A, flag);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void isfinite_async<fp32_t>(int worker_hint, const Tile<fp32_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<fp64_t>(int worker_hint, const Tile<fp64_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<bf16_t>(int worker_hint, const Tile<bf16_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<fp16_t>(int worker_hint, const Tile<fp16_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<fp32_fast_tf32_t>(int worker_hint,
        const Tile<fp32_fast_tf32_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<fp32_fast_fp16_t>(int worker_hint,
        const Tile<fp32_fast_fp16_t> &A, const Tile<bool_t> &flag);

template
void isfinite_async<fp32_fast_bf16_t>(int worker_hint,
        const Tile<fp32_fast_bf16_t> &A, const Tile<bool_t> &flag);

// Explicit instantiation
template
void isfinite<fp32_t>(int worker_hint, const Tile<fp32_t> &A, const Tile<bool_t> &flag);

template
void isfinite<fp64_t>(int worker_hint, const Tile<fp64_t> &A, const Tile<bool_t> &flag);

template
void isfinite<bf16_t>(int worker_hint, const Tile<bf16_t> &A, const Tile<bool_t> &flag);

template
void isfinite<fp16_t>(int worker_hint, const Tile<fp16_t> &A, const Tile<bool_t> &flag);

template
void isfinite<fp32_fast_tf32_t>(int worker_hint,
        const Tile<fp32_fast_tf32_t> &A, const Tile<bool_t> &flag);

template
void isfinite<fp32_fast_fp16_t>(int worker_hint,
        const Tile<fp32_fast_fp16_t> &A, const Tile<bool_t> &flag);

template
void isfinite<fp32_fast_bf16_t>(int worker_hint,
        const Tile<fp32_fast_bf16_t> &A, const Tile<bool_t> &flag);

} // namespace nntile::core
