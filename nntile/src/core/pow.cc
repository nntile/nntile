/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/pow.cc
 * Power operation for Tile<T>
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include "nntile/core/pow.hh"
#include "nntile/kernel_scope.hh"
#include "nntile/nnhaul/ops/pow.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Asynchronous tile-wise power operation
/*! @param[inout] A: Tile for the element-wise power operation
 * */
template<typename T>
void pow_async(int worker_hint, Scalar alpha, Scalar exp, const Tile<T> &A)
{
    nntile::require_kernel_family("pow");
    // Submit task without any arguments checked
    nnhaul::pow.submit<std::tuple<T>>(worker_hint, A.nelems, alpha, exp, A);
}

//! Blocking version of tile-wise power operation
/*! @param[inout] A: Tile for the element-wise power operation
 * */
template<typename T>
void pow(int worker_hint, Scalar alpha, Scalar exp, const Tile<T> &A)
{
    pow_async<T>(worker_hint, alpha, exp, A);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void pow_async<fp32_t>(int worker_hint, Scalar alpha, Scalar exp, const Tile<fp32_t> &A);

template
void pow_async<fp64_t>(int worker_hint, Scalar alpha, Scalar exp, const Tile<fp64_t> &A);

template
void pow_async<bf16_t>(int worker_hint, Scalar alpha, Scalar exp, const Tile<bf16_t> &A);

template
void pow_async<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_tf32_t> &A);

template
void pow_async<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_fp16_t> &A);

template
void pow_async<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_bf16_t> &A);

// Explicit instantiation
template
void pow<fp32_t>(int worker_hint, Scalar alpha, Scalar exp, const Tile<fp32_t> &A);

template
void pow<fp64_t>(int worker_hint, Scalar alpha, Scalar exp, const Tile<fp64_t> &A);

template
void pow<bf16_t>(int worker_hint, Scalar alpha, Scalar exp, const Tile<bf16_t> &A);

template
void pow<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_tf32_t> &A);

template
void pow<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_fp16_t> &A);

template
void pow<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_bf16_t> &A);

} // namespace nntile::core
