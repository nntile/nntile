/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/multiply_slice.cc
 * Tile wrappers for per-element product of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/core/multiply_slice.hh"
#include "nntile/nnhaul/ops/multiply_slice.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

template<typename T>
void multiply_slice_async(int worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index axis)
//! Tile<T> per-element multiplication of a tensor and a broadcasted slice
/*! Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[i,j]
 *
 * @param[in] alpha: Scalar factor
 * @param[in] src: Input slice, that is reshaped into 2D array
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    // Check dimensions
    if(dst.ndim != src.ndim+1)
    {
        throw std::runtime_error("dst.ndim != src.ndim+1");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i-1]");
        }
    }
    // Reshape inputs for simplicity: src -> (m,n), dst -> (m,k,n)
    Index m, n, k;
    m = dst.matrix_shape[axis+1][1];
    n = dst.matrix_shape[axis][0];
    k = dst.shape[axis];
    // Insert corresponding task
    nnhaul::multiply_slice.submit<std::tuple<T>>(worker_hint, m, n, k, alpha, src, dst);
}

template<typename T>
void multiply_slice(int worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst, Index axis)
//! Tile<T> per-element multiplication of a tensor and a broadcasted slice
/*! Blocking version of multiply_slice_async<T>.
 * Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[i,j]
 *
 * @param[in] alpha: Scalar factor
 * @param[in] src: Input slice, that is reshaped into 2D array
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    multiply_slice_async<T>(worker_hint, alpha, src, dst, axis);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation of template
template
void multiply_slice_async<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index axis);

template
void multiply_slice_async<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void multiply_slice_async<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index axis);

template
void multiply_slice_async<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index axis);

template
void multiply_slice_async<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index axis);

template
void multiply_slice_async<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index axis);

template
void multiply_slice_async<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index axis);

// Explicit instantiation of template
template
void multiply_slice<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index axis);

template
void multiply_slice<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void multiply_slice<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index axis);

template
void multiply_slice<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index axis);

template
void multiply_slice<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index axis);

template
void multiply_slice<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index axis);

template
void multiply_slice<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index axis);

} // namespace nntile::core
