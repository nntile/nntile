/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/conv2d_bwd_weight_inplace/cuda.hh
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of weight
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::conv2d_bwd_weight_inplace
{

template<typename T>
void cuda(cudaStream_t stream, Index src1_m, Index src1_n, Index src1_channels,
        Index batch, Index src2_m, Index src2_n, Index stride_m,
        Index stride_n, Index src2_channels, Index offset_m, Index offset_n,
        Scalar alpha, const T *src1, const T *src2, Index dst_m, Index dst_n,
        Index dilation_m, Index dilation_n, Scalar beta, T *dst)
    noexcept;

//! Backward convolution (weight grad) over chunk-backed operands on CUDA
/*! Same operation and per-element arithmetic as cuda() above, with each
 * operand addressed through its device chunk-pointer table. Bitwise the
 * whole-buffer call.
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
        Index stride_m, Index stride_n, Index src2_channels, Index offset_m,
        Index offset_n, Scalar alpha, const T *const *src1_chunks,
        Index nsrc1, const T *const *src2_chunks, Index nsrc2,
        T *const *dst_chunks, Index ndst, Index dst_m, Index dst_n,
        Index dilation_m, Index dilation_n, Scalar beta, Index elems_src1,
        Index elems_src2, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::conv2d_bwd_weight_inplace
