/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/gelutanh.cc
 * Approximate GeLU operation for Tile<T>
 *
 * @version 2.0.0
 * */

#include "nntile/core/gelutanh.hh"
#include "nntile/nnhaul/ops/gelutanh.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Asyncrhonous tile-wise approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise GeLU operation
 * */
template<typename T>
void gelutanh_async(int worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    // Submit task without any arguments checked
    nnhaul::gelutanh.submit<std::tuple<T>>(worker_hint, src.nelems, src, dst);
}

//! Blocking version of tile-wise approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise GeLU operation
 * */
template<typename T>
void gelutanh(int worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    gelutanh_async<T>(worker_hint, src, dst);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void gelutanh_async<fp32_t>(int worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void gelutanh_async<fp32_fast_tf32_t>(int worker_hint, const Tile<fp32_fast_tf32_t> &src,
                                      const Tile<fp32_fast_tf32_t> &dst);

template
void gelutanh_async<fp32_fast_fp16_t>(int worker_hint, const Tile<fp32_fast_fp16_t> &src,
                                const Tile<fp32_fast_fp16_t> &dst);

template
void gelutanh_async<fp32_fast_bf16_t>(int worker_hint, const Tile<fp32_fast_bf16_t> &src,
                                const Tile<fp32_fast_bf16_t> &dst);

template
void gelutanh_async<fp64_t>(int worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void gelutanh_async<bf16_t>(int worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void gelutanh_async<fp16_t>(int worker_hint, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

// Explicit instantiation
template
void gelutanh<fp32_t>(int worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void gelutanh<fp32_fast_tf32_t>(int worker_hint, const Tile<fp32_fast_tf32_t> &src,
                                const Tile<fp32_fast_tf32_t> &dst);

template
void gelutanh<fp32_fast_fp16_t>(int worker_hint, const Tile<fp32_fast_fp16_t> &src,
                                const Tile<fp32_fast_fp16_t> &dst);

template
void gelutanh<fp32_fast_bf16_t>(int worker_hint, const Tile<fp32_fast_bf16_t> &src,
                                const Tile<fp32_fast_bf16_t> &dst);

template
void gelutanh<fp64_t>(int worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void gelutanh<bf16_t>(int worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void gelutanh<fp16_t>(int worker_hint, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

} // namespace nntile::core
