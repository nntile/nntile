/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/multiply_slice/cuda.hh
 * Per-element multiplication of a tensor by a broadcasted slice on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::multiply_slice
{

// Per-element product of a tensor and a broadcasted slice on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src, T *dst)
    noexcept;

//! Per-element product of a tensor and a broadcasted slice on a chunked CUDA
//! buffer
/*! Same operation as cuda() above (dst[i0,l,i1] = alpha * dst[i0,l,i1] *
 * src[i0,i1] on the m-by-k-by-n Fortran layout dst[i0 + m*(l + k*i1)] and
 * the m-by-n layout src[i0 + m*i1]), but both operands' flat element ranges
 * are backed by a chain of equal-size chunk allocations (a shorter tail on
 * the last chunk). The WHOLE region is issued as ONE kernel launch: the two
 * chunk base-pointer tables are host arrays in device-resident memory
 * (built by the driver through the device chunk-ptr ring) whose UVA device
 * views travel as `src_chunks`/`dst_chunks`, and each block stages them into
 * shared memory once. Each element resolves its src chunk from its flat slice
 * offset f = i0 + m*i1 via f / elems_src, f % elems_src and its dst chunk
 * from its flat offset f = i0 + m*(i2 + k*i1) via f / elems_dst,
 * f % elems_dst (shift/mask on the compile-time chunk class). The
 * per-element arithmetic is exactly cuda()'s, so the result is BITWISE the
 * whole-buffer cuda() call. There is no chunk-count cap.
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha: Scalar factor
 * @param[in] src_chunks: Base pointers of src's chunks, in order
 * @param[in] nsrc: Number of src chunks
 * @param[in] elems_src: Elements in a full src chunk (uniform)
 * @param[in] dst_chunks: Base pointers of dst's chunks, in order
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_dst: Elements in a full dst chunk (uniform)
 * */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
    const T *const *src_chunks, Index nsrc, Index elems_src,
    T *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::multiply_slice
