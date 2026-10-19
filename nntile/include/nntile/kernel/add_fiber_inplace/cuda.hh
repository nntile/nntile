/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_fiber_inplace/cuda.hh
 * Per-element addition of a tensor and a broadcasted slice on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::add_fiber_inplace
{

// Per-element addition of a tensor and a broadcasted fiber on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index batch, Scalar alpha,
        const T *src, Scalar beta, T *dst)
    noexcept;

//! Add a chunked broadcasted fiber into a chunked tensor in place (ONE launch)
/*! Same per-element operation as cuda() above (the kernel reads dst even for
 * beta == 0), but src and dst flat ranges are backed by chains of equal-size
 * chunks. One kernel launch over the whole (m, n, k, batch) range; each
 * operand's chunk base-pointer table is a host array in pinned,
 * device-mapped memory (built by the driver through the shared
 * ChunkPtrRing) whose device-ring view travels as the `*_chunks` args, and
 * each block stages it into shared memory once. Every element resolves its
 * chunk from its flat offset (chunk = offset / elems, in-chunk offset =
 * offset % elems; shift/mask on a power-of-two elems). The result is
 * bitwise-equal to cuda()'s. There is no chunk-count cap.
 * */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::add_fiber_inplace
