/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/randn/cuda.hh
 * Randn operation on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::randn
{

template<typename T>
void cuda(
    cudaStream_t stream,
    Index ndim,
    Index nelems,
    unsigned long long seed,
    Scalar mean,
    Scalar stddev,
    const Index *start,
    const Index *shape,
    const Index *underlying_shape,
    T *data,
    const Index *stride
) noexcept;

//! Randn into a chunk-backed, arbitrarily strided output on CUDA
/*! Same per-position PRNG semantics as cuda() above (the value at an
 * output element depends only on the seed and that element's absolute
 * underlying array position), with the destination's strided flat offset
 * resolved through the device chunk-pointer table. Works for any stride
 * geometry, including non-dense (foreign) strides, so one launch covers
 * the output without staging. Bitwise the whole-buffer call.
 *
 * @param[in] stream: CUDA stream
 * @param[in] ndim: Number of dimensions (<= 8)
 * @param[in] nelems: Number of elements in the strided output
 * @param[in] seed: PRNG seed
 * @param[in] mean: Normal distribution mean
 * @param[in] stddev: Normal distribution standard deviation
 * @param[in] start: Starting position in the underlying array
 * @param[in] shape: Strided output shape
 * @param[in] underlying_shape: Shape of the underlying array
 * @param[in] stride: Strides of the output in the underlying array
 * @param[in] dst_chunks: Device-ring view of dst's chunk pointers
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_dst: Elements in a full dst chunk
 * */
template<typename T>
void chunked(
    cudaStream_t stream,
    Index ndim,
    Index nelems,
    unsigned long long seed,
    Scalar mean,
    Scalar stddev,
    const Index *start,
    const Index *shape,
    const Index *underlying_shape,
    const Index *stride,
    T *const *dst_chunks,
    Index ndst,
    Index elems_dst
) noexcept;

} // namespace nntile::kernel::randn
