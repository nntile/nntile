/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/norm.cc
 * Euclidean norm of all elements in a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/norm.hh"
#include "nntile/nnhaul/ops/norm.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

template<typename T>
void norm_async(int worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst)
{
    // Check dimensions
    if(dst.ndim != 0)
    {
        throw std::runtime_error("dst.ndim != 0");
    }
    if(src.nelems == 0)
    {
        throw std::runtime_error("src.nelems == 0");
    }
    // Insert task
    nnhaul::norm.submit<std::tuple<T>>(worker_hint, src.nelems, alpha, src, beta, dst);
}

template<typename T>
void norm(int worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst)
{
    norm_async<T>(worker_hint, alpha, src, beta, dst);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void norm_async<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst);

template
void norm_async<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst);

template
void norm_async<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst);

template
void norm_async<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst);

template
void norm_async<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst);

template
void norm_async<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst);

template
void norm_async<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst);

// Explicit instantiation
template
void norm<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst);

template
void norm<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst);

template
void norm<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst);

template
void norm<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst);

template
void norm<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst);

template
void norm<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst);

template
void norm<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst);

} // namespace nntile::core
