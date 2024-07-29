/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/conv2d/cpu.cc
 * 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/conv2d/cpu.hh"

namespace nntile::kernel::conv2d
{

template <typename T>
void cpu(Index offset_n, Index offset_m, Index batch, Index out_channels,
         Index in_channels, Index padding_n, Index limit_n, Index padding_m,
         Index limit_m, Index src_n, Index src_m, const T *src, Index kernel_n,
         Index kernel_m, const T *kernel, Index dst_n, Index dst_m,
         T *dst) noexcept
//! Compute full discrete linear convolution of two 2-dimensional arrays on
//! CPU. Does not clear initial data, it should be done separately.
/* @param[in] offset_n: Offset alongside first axis
 * @param[in] offset_m: Offset alongside second axis
 * @param[in] batch: Size of batch axes
 * @param[in] out_channels: Size of repeating output tensor channel
 * @param[in] in_channels: Size of repeating input tensor channel
 * @param[in] padding_n: First element of the first axis of the input
 * @param[in] limit_n: Last element of the first axis of the input
 * @param[in] padding_m: First element of the second axis of the input
 * @param[in] limit_m: First element of the second axis of the input
 * @param[in] src_n: Size of the first axis of src array
 * @param[in] src_m: Size of the second axis of src array
 * @param[in] src: Input contiguous src_n-by-src_m array
 * @param[in] kernel_n: Size of the first axis of kernel array
 * @param[in] kernel_m: Size of the second axis of kernel array
 * @param[in] kernel: Input contiguous kernel_n-by-kernel_m array
 * @param[in] dst_n: Size of the first axis of dst array
 * @param[in] dst_m: Size of the second axis of dst array
 * @param[inout] dst: Output contiguous dst_n-by-dst_m array
 * */
{
    using Y = typename T::repr_t;
    auto src_ = reinterpret_cast<const Y *>(src);
    auto kernel_ = reinterpret_cast<const Y *>(kernel);
    auto dst_ = reinterpret_cast<const Y *>(dst);
    Index src_n_end = limit_n;
    if(src_n_end > src_n)
        src_n_end = src_n;
    if(padding_n < 0)
        padding_n = 0;
    Index src_m_end = limit_m;
    if(src_m_end > src_m)
        src_m_end = src_m;
    if(padding_m < 0)
        padding_m = 0;

    for(Index b = 0; b < batch; ++b)
    {
        for(Index oc = 0; oc < out_channels; ++oc)
        {
            for(Index ic = 0; ic < in_channels; ++ic)
            {
                for(Index i1 = padding_n; i1 < src_n_end; ++i1)
                {
                    for(Index j1 = 0; j1 < kernel_n; ++j1)
                    {
                        Index dst_1 = (i1 - padding_n) + j1;
                        if(dst_1 < offset_n || offset_n + dst_n <= dst_1)
                            continue;
                        for(Index i2 = padding_m; i2 < src_m_end; ++i2)
                        {
                            for(Index j2 = 0; j2 < kernel_m; ++j2)
                            {
                                Index dst_2 = (i2 - padding_m) + j2;
                                if(dst_2 < offset_m ||
                                   offset_m + dst_m <= dst_2)
                                    continue;
                                T &dst_val = dst[(dst_1 - offset_n) * dst_m +
                                    (dst_2 - offset_m) + oc * dst_n * dst_m +
                                    b * out_channels * dst_n * dst_m];
                                Y src_val = Y{src[i1 * src_m + i2 +
                                    ic * src_n * src_m +
                                    b * in_channels * src_n * src_m]};
                                Y kernel_val = Y{kernel[j1 * kernel_m + j2 +
                                    oc * kernel_n * kernel_m +
                                    ic * out_channels * kernel_n *
                                    kernel_m]};
                                dst_val = T{Y{dst_val} + src_val*kernel_val};
                            }
                        }
                    }
                }
            }
        }
    }
}

// Explicit instantiation
template void cpu<fp32_t>(Index offset_n, Index offset_m, Index batch,
                          Index out_channels, Index in_channels,
                          Index padding_n, Index limit_n, Index padding_m,
                          Index limit_m, Index src_n, Index src_m,
                          const fp32_t *src, Index kernel_n, Index kernel_m,
                          const fp32_t *kernel, Index dst_n, Index dst_m,
                          fp32_t *dst) noexcept;

template void cpu<fp64_t>(Index offset_n, Index offset_m, Index batch,
                          Index out_channels, Index in_channels,
                          Index padding_n, Index limit_n, Index padding_m,
                          Index limit_m, Index src_n, Index src_m,
                          const fp64_t *src, Index kernel_n, Index kernel_m,
                          const fp64_t *kernel, Index dst_n, Index dst_m,
                          fp64_t *dst) noexcept;

template void cpu<fp32_fast_tf32_t>(Index offset_n, Index offset_m, Index batch,
                          Index out_channels, Index in_channels,
                          Index padding_n, Index limit_n, Index padding_m,
                          Index limit_m, Index src_n, Index src_m,
                          const fp32_fast_tf32_t *src, Index kernel_n, Index kernel_m,
                          const fp32_fast_tf32_t *kernel, Index dst_n, Index dst_m,
                          fp32_fast_tf32_t *dst) noexcept;

template void cpu<bf16_t>(Index offset_n, Index offset_m, Index batch,
                          Index out_channels, Index in_channels,
                          Index padding_n, Index limit_n, Index padding_m,
                          Index limit_m, Index src_n, Index src_m,
                          const bf16_t *src, Index kernel_n, Index kernel_m,
                          const bf16_t *kernel, Index dst_n, Index dst_m,
                          bf16_t *dst) noexcept;

} // namespace nntile::kernel::conv2d
