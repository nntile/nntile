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
#include <iostream>

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
    Index dst_start_m = std::max(offset_m-kernel_m+1, Index(0));
    Index dst_end_m = std::min(offset_m+src_m+kernel_m-1, dst_m);
    Index dst_start_n = std::max(offset_n-kernel_n+1, Index(0));
    Index dst_end_n = std::min(offset_n+src_n+kernel_n-1, dst_n);
    Index src_step = src_n * src_m;
    Index kernel_step = kernel_n * kernel_m;
    for(Index b = 0; b < batch; ++b)
    {
        for(Index oc = 0; oc < out_channels; ++oc)
        {
            for(Index dst_j = 0; dst_j < dst_n; ++dst_j)
            {
                T *dst_slice = dst + ((b*out_channels+oc)*dst_n+dst_j)*dst_m;
                for(Index dst_i = 0; dst_i < dst_m; ++dst_i)
                {
                    // Update within convolution bounds
                    if(dst_i >= dst_start_m and dst_i < dst_end_m and
                            dst_j >= dst_start_n and dst_j < dst_end_n)
                    {
                        Y conv{0.0};
                        Index src_start_m = std::max(dst_i-offset_m,
                                Index(0));
                        Index src_end_m = std::min(dst_i-offset_m+kernel_m,
                                src_m);
                        Index src_start_n = std::max(dst_j-offset_n,
                                Index(0));
                        Index src_end_n = std::min(dst_j-offset_n+kernel_n,
                                src_n);
                        for(Index src_i = src_start_m; src_i < src_end_m;
                                ++src_i)
                        {
                            for(Index src_j = src_start_n; src_j < src_end_n;
                                    ++src_j)
                            {
                                const T *src_slice = src + src_i
                                    + (b*in_channels*src_n+src_j)*src_m;
                                const T *kernel_slice = kernel
                                    + src_i + offset_m - dst_i
                                    + (src_j+offset_n-dst_j)*kernel_m
                                    + oc*in_channels*kernel_step;
                                for(Index ic = 0; ic < in_channels; ++ic)
                                {
                                    Y src_val{src_slice[src_step*ic]};
                                    Y kernel_val{kernel_slice[kernel_step*ic]};
                                    conv += src_val * kernel_val;
                                }
                            }
                        }
                        if(beta == 0.0)
                        {
                            dst_slice[dst_i] = T{alpha * conv};
                        }
                        else
                        {
                            Y old{dst_slice[dst_i]};
                            dst_slice[dst_i] = T{beta*old + alpha*conv};
                        }
                    }
                    // Out of convolution bounds
                    else
                    {
                        if(beta == 0.0)
                        {
                            dst_slice[dst_i] = T{Y{0.0}};
                        }
                        else if(beta != 1.0)
                        {
                            Y old{dst_slice[dst_i]};
                            dst_slice[dst_i] = T{beta * old};
                        }
                    }
                }
            }
        }
    }
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
