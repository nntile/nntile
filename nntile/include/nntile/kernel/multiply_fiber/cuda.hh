/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/multiply_fiber/cuda.hh
 * Per-element multiplication of a tensor by a broadcasted fiber on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::multiply_fiber
{

// Per-element product of a tensor and a broadcasted fiber on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src1, const T *src2, T *dst)
    noexcept;

//! Per-element product of a tensor and a broadcasted fiber on a chunked CUDA
//! buffer
/*! Same operation as cuda() above (dst[i0,l,j] = alpha * src1[l] *
 * src2[i0,l,j] on the m-by-k-by-n Fortran layout dst[i0 + m*(l + k*j)],
 * src2 the same and the k-element fiber operand src1[l]), but all three
 * operands' flat element ranges are backed by a chain of equal-size chunk
 * allocations (a shorter tail on the last chunk). The WHOLE region is issued
 * as ONE kernel launch: the three chunk base-pointer tables are host arrays
 * in device-resident memory (built by the driver through the shared
 * ChunkPtrRing) whose device-ring views travel as the `*_chunks` args, and
 * each block stages them into shared memory once. Each element resolves its
 * src1 chunk from the fiber index l via l / elems_s1, l % elems_s1 and its
 * src2/dst chunks from the flat offset f = i0 + m*(i2 + k*i1) via
 * f / elems_s2, f % elems_s2 and f / elems_dst, f % elems_dst (shift/mask on the
 * compile-time chunk class). The per-element arithmetic is
 * exactly cuda()'s, so the result is BITWISE the whole-buffer cuda() call.
 * There is no chunk-count cap.
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and src1
 * @param[in] alpha: Scalar factor
 * @param[in] src1_chunks: Base pointers of src1's chunks, in order
 * @param[in] ns1: Number of src1 chunks
 * @param[in] elems_s1: Elements in a full src1 chunk (uniform)
 * @param[in] src2_chunks: Base pointers of src2's chunks, in order
 * @param[in] ns2: Number of src2 chunks
 * @param[in] elems_s2: Elements in a full src2 chunk (uniform)
 * @param[in] dst_chunks: Base pointers of dst's chunks, in order
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_dst: Elements in a full dst chunk (uniform)
 * */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
    const T *const *src1_chunks, Index ns1, Index elems_s1,
    const T *const *src2_chunks, Index ns2, Index elems_s2,
    T *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::multiply_fiber
