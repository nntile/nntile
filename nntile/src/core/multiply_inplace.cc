/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/multiply_inplace.cc
 * Per-element multiplication of two Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/multiply_inplace.hh"
#include "nntile/nnhaul/ops/multiply_inplace.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Asynchronous version of tile-wise multiply operation
/*! @param[in] alpha: Scalar multiplier
 * @param[in] src: Input tile for element-wise multiply operation
 * @param[inout] dst: Input and output tile for the multiply operation
 * */
template<typename T>
void multiply_inplace_async(int worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Submit task
    nnhaul::multiply_inplace.submit<std::tuple<T>>(worker_hint, src.nelems, alpha, src,
            dst);
}

//! Blocking version of tile-wise multiply operation
/*! @param[in] alpha: Scalar multiplier
 * @param[in] src: Input tile for element-wise multiply operation
 * @param[inout] dst: Input and output tile for the multiply operation
 * */
template<typename T>
void multiply_inplace(int worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst)
{
    multiply_inplace_async<T>(worker_hint, alpha, src, dst);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void multiply_inplace_async<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void multiply_inplace_async<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void multiply_inplace_async<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void multiply_inplace_async<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void multiply_inplace_async<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void multiply_inplace_async<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

template
void multiply_inplace_async<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

// Explicit instantiation
template
void multiply_inplace<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void multiply_inplace<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void multiply_inplace<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void multiply_inplace<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void multiply_inplace<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void multiply_inplace<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void multiply_inplace<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

} // namespace nntile::core
