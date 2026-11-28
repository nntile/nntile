/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/fill.cc
 * Fill operation for Tile<T>
 *
 * @version 2.0.0
 * */

#include "nntile/core/fill.hh"
#include "nntile/nnhaul/ops/fill.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Asynchronous tile-wise fill operation
/*! @param[inout] A: Tile for the element-wise fill operation
 * */
template<typename T>
void fill_async(int worker_hint, Scalar val, const Tile<T> &A)
{
    // Submit task without any arguments checked
    nnhaul::fill.submit<std::tuple<T>>(worker_hint, A.nelems, val, A);
}

//! Blocking version of tile-wise flll operation
/*! @param[inout] A: Tile for the element-wise fill operation
 * */
template<typename T>
void fill(int worker_hint, Scalar val, const Tile<T> &A)
{
    fill_async<T>(worker_hint, val, A);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void fill_async<fp32_t>(int worker_hint, Scalar val, const Tile<fp32_t> &A);

template
void fill_async<bf16_t>(int worker_hint, Scalar val, const Tile<bf16_t> &A);

template
void fill_async<fp16_t>(int worker_hint, Scalar val, const Tile<fp16_t> &A);

template
void fill_async<fp32_fast_tf32_t>(int worker_hint, Scalar val, const Tile<fp32_fast_tf32_t> &A);

template
void fill_async<fp32_fast_fp16_t>(int worker_hint, Scalar val, const Tile<fp32_fast_fp16_t> &A);

template
void fill_async<fp32_fast_bf16_t>(int worker_hint, Scalar val, const Tile<fp32_fast_bf16_t> &A);

template
void fill_async<fp64_t>(int worker_hint, Scalar val, const Tile<fp64_t> &A);

// Explicit instantiation
template
void fill<fp32_t>(int worker_hint, Scalar val, const Tile<fp32_t> &A);

template
void fill<bf16_t>(int worker_hint, Scalar val, const Tile<bf16_t> &A);

template
void fill<fp16_t>(int worker_hint, Scalar val, const Tile<fp16_t> &A);

template
void fill<fp32_fast_tf32_t>(int worker_hint, Scalar val, const Tile<fp32_fast_tf32_t> &A);

template
void fill<fp32_fast_fp16_t>(int worker_hint, Scalar val, const Tile<fp32_fast_fp16_t> &A);

template
void fill<fp32_fast_bf16_t>(int worker_hint, Scalar val, const Tile<fp32_fast_bf16_t> &A);

template
void fill<fp64_t>(int worker_hint, Scalar val, const Tile<fp64_t> &A);

} // namespace nntile::core
