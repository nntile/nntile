/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/multiply.cc
 * Per-element product of two Tile<T>
 *
 * @version 2.0.0
 * */

#include "nntile/core/multiply.hh"
#include "nntile/nnhaul/ops/multiply.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Asynchronous tile-wise multiply operation
template<typename T>
void multiply_async(int worker_hint, Scalar alpha, const Tile<T> &src1, const Tile<T> &src2,
        const Tile<T> &dst)
{
    // Check shapes
    if(src1.shape != src2.shape)
    {
        throw std::runtime_error("src1.shape != src2.shape");
    }
    if(src1.shape != dst.shape)
    {
        throw std::runtime_error("src1.shape != dst.shape");
    }
    nnhaul::multiply.submit<std::tuple<T>>(worker_hint, src1.nelems, alpha, src1, src2,
            dst);
}

//! Blocking version of tile-wise multiply operation
template<typename T>
void multiply(int worker_hint, Scalar alpha, const Tile<T> &src1, const Tile<T> &src2,
        const Tile<T> &dst)
{
    multiply_async<T>(worker_hint, alpha, src1, src2, dst);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void multiply_async<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src1,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst);

template
void multiply_async<fp32_fast_tf32_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src1,
        const Tile<fp32_fast_tf32_t> &src2,
        const Tile<fp32_fast_tf32_t> &dst);

template
void multiply_async<fp32_fast_fp16_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src1,
        const Tile<fp32_fast_fp16_t> &src2,
        const Tile<fp32_fast_fp16_t> &dst);

template
void multiply_async<fp32_fast_bf16_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src1,
        const Tile<fp32_fast_bf16_t> &src2,
        const Tile<fp32_fast_bf16_t> &dst);

template
void multiply_async<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src1,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst);

template
void multiply_async<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src1,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst);

template
void multiply_async<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src1,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst);

// Explicit instantiation
template
void multiply<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src1,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst);

template
void multiply<fp32_fast_tf32_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src1,
        const Tile<fp32_fast_tf32_t> &src2,
        const Tile<fp32_fast_tf32_t> &dst);

template
void multiply<fp32_fast_fp16_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src1,
        const Tile<fp32_fast_fp16_t> &src2,
        const Tile<fp32_fast_fp16_t> &dst);

template
void multiply<fp32_fast_bf16_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src1,
        const Tile<fp32_fast_bf16_t> &src2,
        const Tile<fp32_fast_bf16_t> &dst);

template
void multiply<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src1,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst);

template
void multiply<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src1,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst);

template
void multiply<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src1,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst);

} // namespace nntile::core
