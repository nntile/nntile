/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/hypot_scalar_inverse.cc
 * hypot_scalar_inverse operation for Tile<T>
 *
 * @version 2.0.0
 * */

#include "nntile/core/hypot_scalar_inverse.hh"
#include "nntile/nnhaul/ops/hypot_scalar_inverse.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

template<typename T>
void hypot_scalar_inverse_async(int worker_hint, Scalar eps, Scalar alpha, const Tile<T> &dst)
{
    nnhaul::hypot_scalar_inverse.submit<std::tuple<T>>(worker_hint, dst.nelems, eps,
            alpha, dst);
}

template<typename T>
void hypot_scalar_inverse(int worker_hint, Scalar eps, Scalar alpha, const Tile<T> &dst)
{
    hypot_scalar_inverse_async<T>(worker_hint, eps, alpha, dst);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation of template
template
void hypot_scalar_inverse_async<fp32_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp32_t> &dst);

template
void hypot_scalar_inverse_async<bf16_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<bf16_t> &dst);

template
void hypot_scalar_inverse_async<fp32_fast_tf32_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &dst);

template
void hypot_scalar_inverse_async<fp32_fast_fp16_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &dst);

template
void hypot_scalar_inverse_async<fp32_fast_bf16_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &dst);

template
void hypot_scalar_inverse_async<fp64_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp64_t> &dst);

template
void hypot_scalar_inverse_async<fp16_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp16_t> &dst);

// Explicit instantiation of template
template
void hypot_scalar_inverse<fp32_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp32_t> &dst);

template
void hypot_scalar_inverse<fp32_fast_tf32_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &dst);

template
void hypot_scalar_inverse<fp32_fast_fp16_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &dst);

template
void hypot_scalar_inverse<fp32_fast_bf16_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &dst);

template
void hypot_scalar_inverse<fp64_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp64_t> &dst);

template
void hypot_scalar_inverse<bf16_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<bf16_t> &dst);

template
void hypot_scalar_inverse<fp16_t>(int worker_hint, Scalar eps, Scalar alpha,
        const Tile<fp16_t> &dst);

} // namespace nntile::core
