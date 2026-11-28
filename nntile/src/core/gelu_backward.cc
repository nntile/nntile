/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/gelu_backward.cc
 * Backward GeLU operation for Tile<T>
 *
 * @version 2.0.0
 * */

#include "nntile/core/gelu_backward.hh"
#include "nntile/nnhaul/ops/gelu_backward.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Asynchronous tile-wise backward GeLU operation
/*! @param[inout] A: Tile for the element-wise backward GeLU operation
 * */
template<typename T>
void gelu_backward_async(int worker_hint, Scalar alpha, const Tile<T> &x, const Tile<T> &dy,
        Scalar beta, const Tile<T> &dx)
{
    // Check shapes
    if(x.shape != dy.shape)
    {
        throw std::runtime_error("x.shape != dy.shape");
    }
    if(x.shape != dx.shape)
    {
        throw std::runtime_error("x.shape != dx.shape");
    }
    // Submit task without any arguments checked
    nnhaul::gelu_backward.submit<std::tuple<T>>(worker_hint, x.nelems, alpha, x, dy, beta, dx);
}

//! Blocking version of tile-wise backward GeLU operation
/*! @param[inout] A: Tile for the element-wise backward GeLU operation
 * */
template<typename T>
void gelu_backward(int worker_hint, Scalar alpha, const Tile<T> &x, const Tile<T> &dy,
        Scalar beta, const Tile<T> &dx)
{
    gelu_backward_async<T>(worker_hint, alpha, x, dy, beta, dx);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void gelu_backward_async<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &x, const Tile<fp32_t> &dy,
        Scalar beta, const Tile<fp32_t> &dx);

template
void gelu_backward_async<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &x,
        const Tile<fp32_fast_tf32_t> &dy, Scalar beta, const Tile<fp32_fast_tf32_t> &dx);

template
void gelu_backward_async<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &x,
        const Tile<fp32_fast_bf16_t> &dy, Scalar beta, const Tile<fp32_fast_bf16_t> &dx);

template
void gelu_backward_async<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &x,
        const Tile<fp32_fast_fp16_t> &dy, Scalar beta, const Tile<fp32_fast_fp16_t> &dx);

template
void gelu_backward_async<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &x,
        const Tile<bf16_t> &dy, Scalar beta, const Tile<bf16_t> &dx);

template
void gelu_backward_async<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &x, const Tile<fp64_t> &dy,
        Scalar beta, const Tile<fp64_t> &dx);

template
void gelu_backward_async<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &x,
        const Tile<fp16_t> &dy, Scalar beta, const Tile<fp16_t> &dx);

// Explicit instantiation
template
void gelu_backward<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &x, const Tile<fp32_t> &dy,
        Scalar beta, const Tile<fp32_t> &dx);

template
void gelu_backward<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &x,
        const Tile<fp32_fast_tf32_t> &dy, Scalar beta, const Tile<fp32_fast_tf32_t> &dx);

template
void gelu_backward<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &x,
        const Tile<fp32_fast_bf16_t> &dy, Scalar beta, const Tile<fp32_fast_bf16_t> &dx);

template
void gelu_backward<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &x,
        const Tile<fp32_fast_fp16_t> &dy, Scalar beta, const Tile<fp32_fast_fp16_t> &dx);

template
void gelu_backward<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &x,
        const Tile<bf16_t> &dy, Scalar beta, const Tile<bf16_t> &dx);

template
void gelu_backward<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &x,
        const Tile<fp16_t> &dy, Scalar beta, const Tile<fp16_t> &dx);

template
void gelu_backward<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &x, const Tile<fp64_t> &dy,
        Scalar beta, const Tile<fp64_t> &dx);

} // namespace nntile::core
