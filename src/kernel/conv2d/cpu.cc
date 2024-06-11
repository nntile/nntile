/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/conv2d/cpu.cc
 * 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-28
 * */

#include "nntile/kernel/conv2d/cpu.hh"
#include <cstdio>
#include <iostream>
namespace nntile
{
namespace kernel
{
namespace conv2d
{

template <typename T>
void cpu(Index offset_n, Index offset_m, Index batch, Index src_n, Index src_m,
         const T *src, Index kernel_n, Index kernel_m, const T *kernel,
         Index dst_n, Index dst_m, T *dst) noexcept
//! Compute full discrete linear convolution of two 2-dimensional arrays on
//! CPU. Does not clear initial data, it should be done separately.
/* @param[in] offset_n: Offset alongside first axis
 * @param[in] offset_m: Offset alongside second axis
 * @param[in] batch: Size of batch axes
 * @param[in] src_n: Size of the first axis of src array
 * @param[in] src_m: Size of the second axis of src array
 * @param[in] src: Input contiguous src_n-by-src_m array
 * @param[in] kernel_n: Size of the first axis of kernel array
 * @param[in] kernel_m: Size of the second axis of kernel array
 * @param[in] kernel: Input contiguous kernel_n-by-kernel_m array
 * @param[in] dst_n: Size of the first axis of dst array
 * @param[in] dst_m: Size of the second axis of dst array
 * @param[out] dst: Output contiguous dst_n-by-dst_m array
 * */
{
    for(Index b = 0; b < batch; ++b)
    {
        for(Index i1 = 0; i1 < src_n; ++i1)
        {
            for(Index j1 = 0; j1 < kernel_n; ++j1)
            {
                Index dst_1 = i1 + j1;
                if(dst_1 < offset_n || offset_n + dst_n <= dst_1)
                    continue;
                for(Index i2 = 0; i2 < src_m; ++i2)
                {
                    for(Index j2 = 0; j2 < kernel_m; ++j2)
                    {
                        Index dst_2 = i2 + j2;
                        if(dst_2 < offset_m || offset_m + dst_m <= dst_2)
                            continue;
                        dst[(dst_1 - offset_n) * dst_m + (dst_2 - offset_m) +
                            b * dst_n * dst_m] +=
                            src[i1 * src_m + i2 + b * src_n * src_m] *
                            kernel[j1 * kernel_m + j2 +
                                   b * kernel_n * kernel_m];
                    }
                }
            }
        }
    }
}

// Explicit instantiation
template void cpu<fp32_t>(Index offset_n, Index offset_m, Index batch,
                          Index src_n, Index src_m, const fp32_t *src,
                          Index kernel_n, Index kernel_m, const fp32_t *kernel,
                          Index dst_n, Index dst_m, fp32_t *dst) noexcept;

template void cpu<fp64_t>(Index offset_n, Index offset_m, Index batch,
                          Index src_n, Index src_m, const fp64_t *src,
                          Index kernel_n, Index kernel_m, const fp64_t *kernel,
                          Index dst_n, Index dst_m, fp64_t *dst) noexcept;

} // namespace conv2d
} // namespace kernel
} // namespace nntile
