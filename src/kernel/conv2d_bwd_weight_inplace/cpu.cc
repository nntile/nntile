/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/conv2d_bwd_weight_inplace/cpu.cc
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of weight
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/conv2d_bwd_weight_inplace/cpu.hh"
#include <algorithm>
#include <iostream>

namespace nntile::kernel::conv2d_bwd_weight_inplace
{

template<typename T>
void cpu(Index src1_m, Index src1_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index src2_channels,
        Index offset_m, Index offset_n, Scalar alpha, const T *src1,
        const T *src2, Index dst_m, Index dst_n, Scalar beta, T *dst)
    noexcept
/*! Backward convolution of WHCN tensors to get grad of weight
 *
 * The following operation is performed:
 *      `dst` = `alpha`*`f(src1, src2)` + `beta`*`dst`,
 * where `f` operation does the following:
 *      `f[i,j,k,l]` = \sum_b \sum_m \sum_n `src1[m,n,k,b]`
 *      * `src2[m + offset_m - i,n + offset_n - i,l,b]`
 *
 * Generally, `src1` represents input of `Conv2d` layer, `src2` represents
 * output grad of `Conv2d` layer and `dst` represents weight grad of `Conv2d`
 * layer.
 *
 * @param[in] src1_m: Size of the thirst axis of `src1` array
 * @param[in] src1_n: Size of the second axis of `src1` array
 * @param[in] src1_channels: Size of the third axis of `src1` array
 * @param[in] batch: Size of the fourth axis of `src1` array
 * @param[in] src2_m: Size of the first axis of `src2` array
 * @param[in] src2_n: Size of the second axis of `src2` array
 * @param[in] src2_channels: Size of the third axis of `src2` array
 * @param[in] offset_m: Convolution offset alongside the first axis
 * @param[in] offset_n: Convolution offset alongside the second axis
 * @param[in] alpha: Scalar multiplier for the convolution operation
 * @param[in] src1: F-contiguous tensor of shape
 *      (`src1_m`,`src1_n`,`src1_channels`,`batch`)
 * @param[in] src2: F-contiguous tensor of shape
 *      (`src2_m`,`src2_n`,`src2_channels`,`batch`)
 * @param[in] dst_m: Size of the first axis of dst array
 * @param[in] dst_n: Size of the second axis of dst array
 * @param[in] beta: Scalar multiplier for initial value of `dst`
 * @param[inout] dst: F-contiguous array of shape
 *      (`dst_m`, `dst_n`, `src1_channels`, `src2_channels`)
 * */
{
    using Y = typename T::repr_t;
    // Let `d` denote a 2-dimensional index within `dst`,
    //     `s1` denote a 2-dim index within `s1`,
    //     `o` denote a 2-dim offset
    // Then, this convolution computes
    //      `conv[d] = sum_s1 src1[s1]*src2[s1-o-d]`
    // And we must satisfy condition
    //      `0 <= s1-o-d < src2.shape`
    // It means all values of `d`, that actually get non-zero `conv[d]` are:
    //      `s1-o-src2.shape < d <= s1-o`
    // To support left border inclusively and right border exclusively:
    //      `1-o-src2.shape <= d < src1.shape-o`
    Index dst_start_m = std::max(1-offset_m-src2_m, Index(0));
    Index dst_end_m = std::min(src1_m-offset_m, dst_m);
    Index dst_start_n = std::max(1-offset_n-src2_n, Index(0));
    Index dst_end_n = std::min(src1_n-offset_n, dst_n);
    Index src1_step = src1_channels * src1_n * src1_m;
    Index src2_step = src2_channels * src2_n * src2_m;
    for(Index oc = 0; oc < src2_channels; ++oc)
    {
        for(Index ic = 0; ic < src1_channels; ++ic)
        {
            for(Index dst_j = 0; dst_j < dst_n; ++dst_j)
            {
                T *dst_slice = dst + ((oc*src1_channels+ic)*dst_n+dst_j)*dst_m;
                for(Index dst_i = 0; dst_i < dst_m; ++dst_i)
                {
                    // Update within convolution bounds
                    if(dst_i >= dst_start_m and dst_i < dst_end_m and
                            dst_j >= dst_start_n and dst_j < dst_end_n)
                    {
                        // Additional variables for Kahan summation rule
                        Y conv{0.0}, c{0}, y, t;
                        // Once again, we must satisfy condition
                        //      `0 <= s1-o-d < src2.shape`
                        // Therefore, indices `s1` are bound as follows:
                        //      `o+d <= s1 < o+d+src2.shape`
                        Index src1_start_m = std::max(offset_m+dst_i,
                                Index(0));
                        Index src1_end_m = std::min(offset_m+src2_m+dst_i,
                                src1_m);
                        Index src1_start_n = std::max(offset_n+dst_j,
                                Index(0));
                        Index src1_end_n = std::min(offset_n+src2_n+dst_j,
                                src1_n);
                        for(Index src1_i = src1_start_m; src1_i < src1_end_m;
                                ++src1_i)
                        {
                            for(Index src1_j = src1_start_n;
                                    src1_j < src1_end_n; ++src1_j)
                            {
                                const T *src1_slice = src1 + src1_i
                                    + (ic*src1_n+src1_j)*src1_m;
                                // Slice of `src2[s1-o-d]`
                                const T *src2_slice = src2
                                    + src1_i - offset_m - dst_i
                                    + (oc*src2_n+src1_j-offset_n-dst_j)*src2_m;
                                for(Index b = 0; b < batch; ++b)
                                {
                                    Y src1_val{src1_slice[src1_step*b]};
                                    Y src2_val{src2_slice[src2_step*b]};
                                    // Kahan rule to get the following sum
                                    // conv += src1_val * src2_val
                                    y = src1_val*src2_val - c;
                                    t = conv + y;
                                    c = (t-conv) - y;
                                    conv = t;
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
                            dst_slice[dst_i] = T{(beta*old - alpha*c)
                                + alpha*conv};
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
void cpu<bf16_t>(Index src1_m, Index src1_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index src2_channels, Index offset_m,
        Index offset_n, Scalar alpha, const bf16_t *src1, const bf16_t *src2,
        Index dst_m, Index dst_n, Scalar beta, bf16_t *dst)
    noexcept;

template
void cpu<fp32_t>(Index src1_m, Index src1_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index src2_channels, Index offset_m,
        Index offset_n, Scalar alpha, const fp32_t *src1, const fp32_t *src2,
        Index dst_m, Index dst_n, Scalar beta, fp32_t *dst)
    noexcept;

template
void cpu<fp32_fast_tf32_t>(Index src1_m, Index src1_n, Index src1_channels,
        Index batch, Index src2_m, Index src2_n, Index src2_channels,
        Index offset_m, Index offset_n, Scalar alpha,
        const fp32_fast_tf32_t *src1, const fp32_fast_tf32_t *src2,
        Index dst_m, Index dst_n, Scalar beta, fp32_fast_tf32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index src1_m, Index src1_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index src2_channels, Index offset_m,
        Index offset_n, Scalar alpha, const fp64_t *src1, const fp64_t *src2,
        Index dst_m, Index dst_n, Scalar beta, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::conv2d_bwd_weight_inplace
