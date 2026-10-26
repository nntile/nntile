/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/kernel/swap_two_axes/cuda.hh
 * Swap axes 1 and 3 in a 5D buffer on CUDA.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::swap_two_axes
{

template<typename T>
void cuda(
    cudaStream_t stream,
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const T *src,
    T *dst) noexcept;

//! Swap axes 1 and 3 of a chunk-backed 5D buffer on CUDA
/*! Same operation as cuda() above, but both operands' flat element ranges
 * are backed by a chain of equal-size chunk allocations (a shorter exact
 * tail on the last chunk). The whole swap is issued as ONE kernel launch:
 * the chunk base-pointer tables are read from the device chunk-pointer
 * ring, and each source/destination flat offset is resolved to
 * (chunk, offset) with the power-of-two class shift/mask. Elementwise and
 * deterministic, so the result is bitwise the whole-buffer cuda() call.
 *
 * @param[in] stream: CUDA stream
 * @param[in] d0,d1,d2,d3,d4: buffer extents
 * @param[in] src_chunks: Device-ring view of src's chunk base pointers
 * @param[in] nsrc: Number of src chunks
 * @param[in] dst_chunks: Device-ring view of dst's chunk base pointers
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_src: Elements in a full src chunk (uniform)
 * @param[in] elems_dst: Elements in a full dst chunk (uniform)
 * */
template<typename T>
void chunked(
    cudaStream_t stream,
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const T *const *src_chunks,
    Index nsrc,
    T *const *dst_chunks,
    Index ndst,
    Index elems_src,
    Index elems_dst) noexcept;

} // namespace nntile::kernel::swap_two_axes
