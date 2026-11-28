/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/sqrt.cc
 * Sqrt operation for Tile<T>
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include "nntile/core/sqrt.hh"
#include "nntile/kernel_scope.hh"
#include "nntile/nnhaul/ops/sqrt.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Asynchronous tile-wise sqrt operation
/*! @param[inout] A: Tile for the element-wise sqrt operation
 * */
template<typename T>
void sqrt_async(int worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    nntile::require_kernel_family("sqrt");
    // Submit task without any arguments checked
    nnhaul::sqrt.submit<std::tuple<T>>(worker_hint, src.nelems, src, dst);
}

//! Blocking version of tile-wise sqrt operation
/*! @param[inout] A: Tile for the element-wise sqrt operation
 * */
template<typename T>
void sqrt(int worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    sqrt_async<T>(worker_hint, src, dst);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void sqrt_async<fp32_t>(int worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void sqrt_async<fp64_t>(int worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void sqrt_async<bf16_t>(int worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void sqrt_async<fp32_fast_tf32_t>(int worker_hint, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void sqrt_async<fp32_fast_fp16_t>(int worker_hint, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void sqrt_async<fp32_fast_bf16_t>(int worker_hint, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

// Explicit instantiation
template
void sqrt<fp32_t>(int worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void sqrt<fp64_t>(int worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void sqrt<bf16_t>(int worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void sqrt<fp32_fast_tf32_t>(int worker_hint, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void sqrt<fp32_fast_fp16_t>(int worker_hint, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void sqrt<fp32_fast_bf16_t>(int worker_hint, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

} // namespace nntile::core
