/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/conv2d_bwd_input_inplace/cuda.hh
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of input
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::conv2d_bwd_input_inplace
{

template<typename T>
void cuda(cudaStream_t stream, Index src1_m, Index src1_n, Index stride_m,
        Index stride_n, Index src1_channels, Index batch, Index src2_m,
        Index src2_n, Index dilation_m, Index dilation_n, Index dst_channels,
        Index offset_m, Index offset_n, Scalar alpha, const T *src1,
        const T *src2, Index dst_m, Index dst_n, Scalar beta, T *dst)
    noexcept;

} // namespace nntile::kernel::conv2d_bwd_input_inplace
