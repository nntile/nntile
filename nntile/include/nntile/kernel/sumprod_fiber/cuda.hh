/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sumprod_fiber/cuda.hh
 * Sums over slices into a fiber of a product of buffers on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::sumprod_fiber
{

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src1, const T *src2, Scalar beta, T *dst)
    noexcept;

//! Sum fiber products of chunked tensors into a fiber on CUDA
/*! Same operation as cuda() above, but every operand's flat element range
 * is backed by a chain of equal-size chunk allocations (a shorter tail on
 * the last chunk). The whole reduction is issued as ONE kernel launch: the
 * three chunk base-pointer tables are host arrays in device-resident
 * memory (built by the driver through the device chunk-ptr ring) whose UVA
 * device views travel as the `*_chunks` args, and each block stages them into
 * shared memory once. Every element derives its chunk index from its own flat
 * offset (chunk = offset / elems, in-chunk offset = offset % elems; a
 * shift/mask on the compile-time chunk class). The kernel reproduces, for each of
 * cuda()'s branches, that branch's thread geometry and accumulation order (the
 * m==1 kernel's column-strided per-thread product sum and shared tree
 * reduction; the general kernel's per-thread fiber walk and shared atomicAdd
 * reduction), so the result equals cuda()'s up to the same atomic ordering
 * nondeterminism the general branch already has - and is bitwise cuda()'s on
 * the m==1 branch, which has no atomics. There is no chunk-count cap.
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of src1 and src2
 * @param[in] n: Size of the last mode of src1 and src2
 * @param[in] k: Size of the middle mode and of the only mode of dst
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1_chunks: Base pointers of src1's chunks, in order
 * @param[in] nsrc1: Number of src1 chunks
 * @param[in] src2_chunks: Base pointers of src2's chunks, in order
 * @param[in] nsrc2: Number of src2 chunks
 * @param[in] beta: Scaling factor for dst
 * @param[in] dst_chunks: Base pointers of dst's chunks, in order
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_src1: Elements in a full src1 chunk (uniform)
 * @param[in] elems_src2: Elements in a full src2 chunk (uniform)
 * @param[in] elems_dst: Elements in a full dst chunk (uniform)
 * */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T* const* src1_chunks, Index nsrc1,
        const T* const* src2_chunks, Index nsrc2, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src1, Index elems_src2,
        Index elems_dst)
    noexcept;

} // namespace nntile::kernel::sumprod_fiber
