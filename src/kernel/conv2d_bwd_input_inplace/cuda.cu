/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/conv2d_bwd_input_inplace/cuda.cu
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of input
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/conv2d_bwd_input_inplace/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::conv2d_bwd_input_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index src1_m, Index src1_n, Index stride_m, Index stride_n,
        Index src1_channels, Index batch, Index src2_m, Index src2_n,
        Index dilation_m, Index dilation_n, Index dst_channels, Index offset_m,
        Index offset_n, Scalar alpha, const T *src1, const T *src2,
        Index dst_m, Index dst_n, Scalar beta, T *dst)
/*! Backward convolution of WHCN tensors to get grad of input
 *
 * The following operation is performed:
 *      `dst` = `alpha`*`f(src1, src2)` + `beta`*`dst`,
 * where `f` operation does the following:
 *      `f[i,j,k,b]` = \sum_l \sum_m \sum_n `src1[m,n,l,b]`
 *      * `src2[(i + offset_m - stride_m*m) / dilation_m,
 *              (j + offset_n - stride_n*n) / dilation_n,k,l]`
 * with `(i + offset_m - stride_m*m) % dilation_m == 0`
 * and `(j + offset_n - stride_n*n) % dilation_n == 0`
 *
 * Generally, `src1` represents output grad of `Conv2d` layer, `src2`
 * represents kernel of `Conv2d` layer and `dst` represents input grad of
 * `Conv2d` layer.
 *
 * @param[in] src1_m: Size of the first axis of `src1` array
 * @param[in] src1_n: Size of the second axis of `src1` array
 * @param[in] stride_m: Step of the first axis of `src1` array
 * @param[in] stride_n: Step of the first axis of `src1` array
 * @param[in] src1_channels: Size of the third axis of `src1` array
 * @param[in] batch: Size of the fourth axis of `src1` array
 * @param[in] src2_m: Size of the first axis of `src2` array
 * @param[in] src2_n: Size of the second axis of `src2` array
 * @param[in] dilation_m: dilation effect of kernel (`src2`) array
 * @param[in] dilation_n: dilation effect of kernel (`src2`) array
 * @param[in] dst_channels: Size of the third axis of `dst` array
 * @param[in] offset_m: Convolution offset alongside the first axis
 * @param[in] offset_n: Convolution offset alongside the second axis
 * @param[in] alpha: Scalar multiplier for the convolution operation
 * @param[in] src1: F-contiguous tensor of shape
 *      (`src1_m`,`src1_n`,`src1_channels`,`batch`)
 * @param[in] src2: F-contiguous tensor of shape
 *      (`src2_m`,`src2_n`,`dst_channels`,`src1_channels`)
 * @param[in] dst_m: Size of the first axis of dst array
 * @param[in] dst_n: Size of the second axis of dst array
 * @param[in] beta: Scalar multiplier for initial value of `dst`
 * @param[inout] dst: F-contiguous array of shape
 *      (`dst_m`, `dst_n`, `dst_channels`, `batch`)
 * */
{
    using Y = typename T::repr_t;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Let `d` denote a 2-dimensional index within `dst`,
    //     `s1` denote a 2-dim index within `s1`,
    //     `o` denote a 2-dim offset from `dst` to `src1`
    // Then, this convolution computes
    //      `conv[d] = sum_s1 src1[s1]*src2[(d+o-stride*s1)/dilation]`
    // And we must satisfy condition
    //      `0 <= d+o-stride*s1 <= dilation*(src2.shape-1)`
    // It means all values of `d`, that actually get non-zero `conv[d]` are:
    //      `stride*s1-o <= d <= dilation*(src2.shape-1)+stride*s1-o`
    // Therefore, index `d` is bound as follows:
    //      `-o <= d < dilation*(src2.shape-1)+stride*(src1.shape-1)-o+1`
    Index dst_i = i % dst_m;
    i = i / dst_m;
    Index dst_j = i % dst_n;
    i = i / dst_n;
    Index oc = i % dst_channels;
    Index b = i / dst_channels;
    // Early exit if such destination is out of bounds
    if(b >= batch)
    {
        return;
    }
    Index dst_start_m = ::max(-offset_m, Index(0));
    Index dst_end_m = ::min(
            dilation_m*(src2_m-1)+stride_m*(src1_m-1)-offset_m+1,
            dst_m);
    Index dst_start_n = ::max(-offset_n, Index(0));
    Index dst_end_n = ::min(
            dilation_n*(src2_n-1)+stride_n*(src1_n-1)-offset_n+1,
            dst_n);
    T *dst_val = dst + ((b*dst_channels+oc)*dst_n+dst_j)*dst_m + dst_i;
    if(dst_i >= dst_start_m and dst_i < dst_end_m and
            dst_j >= dst_start_n and dst_j < dst_end_n)
    {
        Index src1_step = src1_n * src1_m;
        Index src2_oc_step = src2_n * src2_m;
        Index src2_ic_step = src2_oc_step * dst_channels;
        // Additional variables for Kahan summation rule
        Y conv{0.0}, c{0}, y, t;
        // Once again, we must satisfy condition
        //      `0 <= d+o-stride*s1 <= dilation*(src2.shape-1)`
        // with `(d+o-s1*stride) % dilation == 0`
        // Finally, we get
        //      `s1 >= ceil((d+o-dilation*(src2.shape-1)
        //          / stride)`
        //      `s1 < floor((d+o+stride)/stride)`
        Index src1_start_m = (dst_i+offset_m-dilation_m*(src2_m-1)+stride_m-1)
            / stride_m;
        src1_start_m = ::max(src1_start_m, Index(0));
        Index src1_end_m = ::min((dst_i+offset_m+stride_m) / stride_m, src1_m);
        Index src1_start_n = (dst_j+offset_n-dilation_n*(src2_n-1)+stride_n-1)
            / stride_n;
        src1_start_n = ::max(src1_start_n, Index(0));
        Index src1_end_n = ::min((dst_j+offset_n+stride_n) / stride_n, src1_n);
        for(Index src1_i = src1_start_m; src1_i < src1_end_m;
                ++src1_i)
        {
            Index tmp_i = dst_i+offset_m-stride_m*src1_i;
            // Ensure `(d+o-stride*s1) % dilation == 0`
            if(tmp_i % dilation_m != 0)
            {
                continue;
            }
            for(Index src1_j = src1_start_n;
                    src1_j < src1_end_n; ++src1_j)
            {
                Index tmp_j = dst_j+offset_n-stride_n*src1_j;
                // Ensure `(d+o-stride*s1) % dilation == 0`
                if(tmp_j % dilation_n != 0)
                {
                    continue;
                }
                const T *src1_slice = src1 + src1_i
                    + (b*src1_channels*src1_n+src1_j)*src1_m;
                // Slice of `src2[(d+o-stride*s1)/dilation]`
                const T *src2_slice = src2
                    + tmp_i/dilation_m
                    + tmp_j/dilation_n*src2_m
                    + oc*src2_oc_step;
                for(Index ic = 0; ic < src1_channels; ++ic)
                {
                    Y src1_val{src1_slice[src1_step*ic]};
                    Y src2_val{src2_slice[src2_ic_step*ic]};
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
            *dst_val = T{alpha * conv};
        }
        else
        {
            Y old{*dst_val};
            *dst_val = T{(beta*old - alpha*c) + alpha*conv};
        }
    }
    // Out of convolution bounds
    else
    {
        if(beta == 0.0)
        {
            *dst_val = T{Y{0.0}};
        }
        else if(beta != 1.0)
        {
            Y old{*dst_val};
            *dst_val = T{beta * old};
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index src1_m, Index src1_n, Index stride_m,
        Index stride_n, Index src1_channels, Index batch, Index src2_m,
        Index src2_n, Index dilation_m, Index dilation_n, Index dst_channels,
        Index offset_m, Index offset_n, Scalar alpha, const T *src1,
        const T *src2, Index dst_m, Index dst_n, Scalar beta, T *dst)
    noexcept
{
    int nelems_dst = dst_m * dst_n * dst_channels * batch;
    dim3 blocks((nelems_dst+255)/256), threads(256);
    cuda_kernel<T><<<blocks, threads, 0, stream>>>(src1_m, src1_n, stride_m,
            stride_n, src1_channels, batch, src2_m, src2_n, dilation_m,
            dilation_n, dst_channels, offset_m, offset_n, alpha, src1, src2,
            dst_m, dst_n, beta, dst);
}

// Explicit instantiation
template
void cuda<bf16_t>(cudaStream_t stream, Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const bf16_t *src1, const bf16_t *src2, Index dst_m, Index dst_n,
        Scalar beta, bf16_t *dst)
    noexcept;

template
void cuda<fp32_t>(cudaStream_t stream, Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const fp32_t *src1, const fp32_t *src2, Index dst_m, Index dst_n,
        Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const fp32_fast_tf32_t *src1, const fp32_fast_tf32_t *src2,
        Index dst_m, Index dst_n, Scalar beta,
        fp32_fast_tf32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const fp64_t *src1, const fp64_t *src2, Index dst_m, Index dst_n,
        Scalar beta, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::conv2d_bwd_input_inplace
