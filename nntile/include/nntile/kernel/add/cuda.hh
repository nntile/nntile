/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add/cuda.hh
 * Add operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::add
{

// Apply add for buffers on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *src1,
        Scalar beta, const T *src2, T *dst)
    noexcept;

//! Add two chunked buffers on CUDA (ONE launch, chunk tables via UVA)
/*! Same per-element operation as cuda() above, but each operand's flat
 * element range is backed by a chain of equal-size chunk allocations (a
 * shorter tail on the last chunk). The whole addition is issued as ONE
 * kernel launch: each operand's chunk base-pointer table is a host array in
 * device-resident memory (built by the driver through the shared
 * ChunkPtrRing) whose device-ring view travels as `src1_chunks`, and each
 * block stages it into shared memory once. Every element derives its chunk
 * index from its own flat offset (chunk = offset / elems, in-chunk offset =
 * offset % elems; a shift/mask on the compile-time chunk class; a single-chunk operand uses the sentinel shift 62 so there is no runtime-divisor path). The chunked
 * result is bitwise-equal to cuda()'s because every element computes the
 * same expression on the same values. There is no chunk-count cap.
 * */
template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        const T* const* src1_chunks, Index nsrc1, Scalar beta,
        const T* const* src2_chunks, Index nsrc2, T* const* dst_chunks,
        Index ndst, Index elems_src1, Index elems_src2, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::add
