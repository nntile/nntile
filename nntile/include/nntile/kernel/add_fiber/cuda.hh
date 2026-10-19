/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_fiber/cuda.hh
 * Per-element addition of a tensor and a broadcasted slice on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::add_fiber
{

// Per-element addition of a tensor and a broadcasted fiber on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index batch, Scalar alpha,
        const T *src1, Scalar beta, const T *src2, T *dst)
    noexcept;

//! Add a chunked tensor and a broadcasted fiber on CUDA (ONE launch)
/*! Same per-element operation as cuda() above, but src1, src2 and dst flat
 * ranges are backed by chains of equal-size chunks. One kernel launch over
 * the whole (m, n, k, batch) range; each operand's chunk base-pointer table
 * is a host array in device-resident memory (built by the driver
 * through the device chunk-ptr ring) whose device-ring view travels as the
 * `*_chunks` args, and each block stages it into shared memory once. Every
 * element resolves its chunk from its flat offset (chunk = offset / elems,
 * in-chunk offset = offset % elems; shift/mask on a power-of-two elems).
 * Element arithmetic is identical, so the result is bitwise-equal to cuda()'s.
 * There is no chunk-count cap.
 * */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T* const* src1_chunks, Index nsrc1, Scalar beta,
        const T* const* src2_chunks, Index nsrc2, T* const* dst_chunks,
        Index ndst, Index elems_src1, Index elems_src2, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::add_fiber
