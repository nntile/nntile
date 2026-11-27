/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/gelutanh_inplace.cc
 * Approximate GeLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/gelutanh_inplace.hh"
#include "nntile/nnhaul/ops/gelutanh_inplace.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Asyncrhonous tile-wise approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise GeLU operation
 * */
template<typename T>
void gelutanh_inplace_async(int worker_hint, const Tile<T> &A)
{
    // Submit task without any arguments checked
    nnhaul::gelutanh_inplace.submit<std::tuple<T>>(worker_hint, A.nelems, A);
}

//! Blocking version of tile-wise approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise GeLU operation
 * */
template<typename T>
void gelutanh_inplace(int worker_hint, const Tile<T> &A)
{
    gelutanh_inplace_async<T>(worker_hint, A);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void gelutanh_inplace_async<fp32_t>(int worker_hint, const Tile<fp32_t> &A);

template
void gelutanh_inplace_async<fp64_t>(int worker_hint, const Tile<fp64_t> &A);

template
void gelutanh_inplace_async<fp32_fast_tf32_t>(int worker_hint, const Tile<fp32_fast_tf32_t> &A);

template
void gelutanh_inplace_async<fp32_fast_fp16_t>(int worker_hint, const Tile<fp32_fast_fp16_t> &A);

template
void gelutanh_inplace_async<fp32_fast_bf16_t>(int worker_hint, const Tile<fp32_fast_bf16_t> &A);

template
void gelutanh_inplace_async<bf16_t>(int worker_hint, const Tile<bf16_t> &A);

// Explicit instantiation
template
void gelutanh_inplace<fp32_t>(int worker_hint, const Tile<fp32_t> &A);

template
void gelutanh_inplace<fp64_t>(int worker_hint, const Tile<fp64_t> &A);

template
void gelutanh_inplace<fp32_fast_tf32_t>(int worker_hint, const Tile<fp32_fast_tf32_t> &A);

template
void gelutanh_inplace<fp32_fast_fp16_t>(int worker_hint, const Tile<fp32_fast_fp16_t> &A);

template
void gelutanh_inplace<fp32_fast_bf16_t>(int worker_hint, const Tile<fp32_fast_bf16_t> &A);

template
void gelutanh_inplace<bf16_t>(int worker_hint, const Tile<bf16_t> &A);

} // namespace nntile::core
