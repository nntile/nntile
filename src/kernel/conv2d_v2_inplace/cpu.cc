/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/conv2d_v2_inplace/cpu.cc
 * 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/conv2d_v2_inplace/cpu.hh"
#include <algorithm>

namespace nntile::kernel::conv2d_v2_inplace
{

template<typename T>
void cpu(Index src_m, Index src_n, Index in_channels, Index batch,
        Index offset_m, Index offset_n, Scalar alpha, const T *src,
        Index kernel_m, Index kernel_n, Index out_channels, const T *kernel,
        Index dst_m, Index dst_n, Scalar beta, T *dst)
    noexcept
/*! Compute full discrete linear convolution of two 2-dimensional arrays
 *
 * @param[in] src_m: Size of the first axis of src array
 * @param[in] src_n: Size of the second axis of src array
 * @param[in] in_channels: Size of repeating input tensor channel
 * @param[in] batch: Size of batch axes
 * @param[in] offset_m: Offset alongside first axis
 * @param[in] offset_n: Offset alongside second axis
 * @param[in] alpha: Scalar multiplier for convolution operation
 * @param[in] src: Input contiguous (src_n, src_m, out_channels, batch)
 * @param[in] kernel_m: Size of the first axis of kernel array
 * @param[in] kernel_n: Size of the second axis of kernel array
 * @param[in] out_channels: Size of repeating output tensor channel
 * @param[in] kernel: Input contiguous kernel_n-by-kernel_m array
 * @param[in] dst_m: Size of the first axis of dst array
 * @param[in] dst_n: Size of the second axis of dst array
 * @param[in] beta: Scalar multiplier for initial dst
 * @param[inout] dst: Output contiguous dst_n-by-dst_m array
 * */
{
    using Y = typename T::repr_t;
    Index src_start_m = std::max(offset_m, Index(0));
    Index src_end_m = std::min(offset_m+src_m+kernel_m, dst_m);
    Index src_start_n = std::max(offset_n, Index(0));
    Index src_end_n = std::min(offset_n+src_n+kernel_n, dst_n);
//    for(Index b = 0; b < batch; ++b)
//    {
//        for(Index oc = 0; oc < out_channels; ++oc)
//        {
//            for(Index ic = 0; ic < in_channels; ++ic)
//            {
//                for(Index i1 = padding_n; i1 < src_n_end; ++i1)
//                {
//                    for(Index j1 = 0; j1 < kernel_n; ++j1)
//                    {
//                        Index dst_1 = (i1 - padding_n) + j1;
//                        if(dst_1 < offset_n || offset_n + dst_n <= dst_1)
//                            continue;
//                        for(Index i2 = padding_m; i2 < src_m_end; ++i2)
//                        {
//                            for(Index j2 = 0; j2 < kernel_m; ++j2)
//                            {
//                                Index dst_2 = (i2 - padding_m) + j2;
//                                if(dst_2 < offset_m ||
//                                   offset_m + dst_m <= dst_2)
//                                    continue;
//                                T &dst_val = dst[(dst_1 - offset_n) * dst_m +
//                                    (dst_2 - offset_m) + oc * dst_n * dst_m +
//                                    b * out_channels * dst_n * dst_m];
//                                Y src_val = Y{src[i1 * src_m + i2 +
//                                    ic * src_n * src_m +
//                                    b * in_channels * src_n * src_m]};
//                                Y kernel_val = Y{kernel[j1 * kernel_m + j2 +
//                                    oc * kernel_n * kernel_m +
//                                    ic * out_channels * kernel_n *
//                                    kernel_m]};
//                                dst_val = T{Y{dst_val} + src_val*kernel_val};
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index src_m, Index src_n, Index in_channels, Index batch,
        Index offset_m, Index offset_n, Scalar alpha, const fp32_t *src,
        Index kernel_m, Index kernel_n, Index out_channels,
        const fp32_t *kernel, Index dst_m, Index dst_n, Scalar beta,
        fp32_t *dst) noexcept;

template
void cpu<fp64_t>(Index src_m, Index src_n, Index in_channels, Index batch,
        Index offset_m, Index offset_n, Scalar alpha, const fp64_t *src,
        Index kernel_m, Index kernel_n, Index out_channels,
        const fp64_t *kernel, Index dst_m, Index dst_n, Scalar beta,
        fp64_t *dst) noexcept;

} // namespace nntile::kernel::conv2d
