/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/conv2d_inplace/cuda.hh
 * Forward 2D-Convolution of two tensors in WHCN format
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::conv2d_inplace
{

template<typename T>
void cuda(cudaStream_t stream, Index src1_m, Index src1_n, Index src1_channels,
        Index batch, Index src2_m, Index src2_n, Index dilation_m,
        Index dilation_n, Index dst_channels, Index offset_m, Index offset_n,
        Scalar alpha, const T *src1, const T *src2, Index dst_m, Index dst_n,
        Index stride_m, Index stride_n, Scalar beta, T *dst)
    noexcept;

//! Forward 2D-convolution over chunk-backed operands on CUDA
/*! Same operation as cuda() above, but each operand's flat element range
 * is backed by a chain of equal-size chunk allocations. The whole
 * destination range is processed by ONE launch; every src1/src2/dst flat
 * offset is resolved through the device chunk-pointer ring with the
 * power-of-two class shift/mask. The per-element arithmetic (index bounds,
 * loop order, Kahan accumulation, alpha/beta handling) is identical to the
 * whole-buffer kernel, so the result is bitwise.
 *
 * @param[in] stream: CUDA stream
 * @param[in] src1_chunks: Device-ring view of src1's chunk pointers
 * @param[in] nsrc1: Number of src1 chunks
 * @param[in] src2_chunks: Device-ring view of src2's chunk pointers
 * @param[in] nsrc2: Number of src2 chunks
 * @param[in] dst_chunks: Device-ring view of dst's chunk pointers
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_src1: Elements in a full src1 chunk
 * @param[in] elems_src2: Elements in a full src2 chunk
 * @param[in] elems_dst: Elements in a full dst chunk
 * */
template<typename T>
void chunked(cudaStream_t stream, Index src1_m, Index src1_n,
        Index src1_channels, Index batch, Index src2_m, Index src2_n,
        Index dilation_m, Index dilation_n, Index dst_channels, Index offset_m,
        Index offset_n, Scalar alpha, const T *const *src1_chunks,
        Index nsrc1, const T *const *src2_chunks, Index nsrc2,
        T *const *dst_chunks, Index ndst, Index dst_m, Index dst_n,
        Index stride_m, Index stride_n, Scalar beta, Index elems_src1,
        Index elems_src2, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::conv2d_inplace
