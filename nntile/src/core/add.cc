/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/add.cc
 * Add operation for two Tile<T>'s
 *
 * @version 2.0.0
 * */

#include "nntile/core/add.hh"
#include "nntile/nnhaul/ops/add.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Tile-wise add operation
template<typename T>
void add_async(int worker_hint, Scalar alpha, const Tile<T> &src1, Scalar beta,
        const Tile<T> &src2, const Tile<T> &dst)
{
    // Check dimensions
    if(dst.ndim != src1.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    if(dst.ndim != src2.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src1.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src1.shape[i]");
        }
    }
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(src2.shape[i] != src1.shape[i])
        {
            throw std::runtime_error("src2.shape[i] != src1.shape[i]");
        }
    }
    // Do nothing if alpha is zero and beta is one
    if(alpha == 0.0 && beta == 1.0)
    {
        return;
    }
    // Insert corresponding task
    nnhaul::add.submit<std::tuple<T>>(worker_hint,
        src1.nelems, alpha, src1, beta, src2, dst);
}

//! Tile-wise add operation
template<typename T>
void add(int worker_hint, Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2,
        const Tile<T> &dst)
{
    add_async<T>(worker_hint, alpha, src1, beta, src2, dst);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation of template
template
void add_async<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src1, Scalar beta,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst);

template
void add_async<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src1, Scalar beta,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst);

template
void add_async<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src1, Scalar beta,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst);

template
void add_async<fp32_fast_tf32_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tile<fp32_fast_tf32_t> &src2, const Tile<fp32_fast_tf32_t> &dst);

template
void add_async<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src1,
        Scalar beta, const Tile<fp32_fast_fp16_t> &src2,
        const Tile<fp32_fast_fp16_t> &dst);

template
void add_async<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src1,
        Scalar beta, const Tile<fp32_fast_bf16_t> &src2,
        const Tile<fp32_fast_bf16_t> &dst);

template
void add_async<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src1, Scalar beta,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst);

// Explicit instantiation of template
template
void add<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src1, Scalar beta,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst);

template
void add<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src1, Scalar beta,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst);

template
void add<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src1, Scalar beta,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst);

template
void add<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src1,
        Scalar beta, const Tile<fp32_fast_tf32_t> &src2,
        const Tile<fp32_fast_tf32_t> &dst);

template
void add<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src1,
        Scalar beta, const Tile<fp32_fast_fp16_t> &src2,
        const Tile<fp32_fast_fp16_t> &dst);

template
void add<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src1,
        Scalar beta, const Tile<fp32_fast_bf16_t> &src2,
        const Tile<fp32_fast_bf16_t> &dst);

template
void add<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src1, Scalar beta,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst);

} // namespace nntile::core
