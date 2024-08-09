/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/conv2d_bwd_weight_inplace/cuda.cu
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of weight
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/conv2d_bwd_weight_inplace/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::conv2d_bwd_weight_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index src1_m, Index src1_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index stride_m, Index stride_n,
        Index src2_channels, Index offset_m, Index offset_n, Scalar alpha,
        const T *src1, const T *src2, Index dst_m, Index dst_n,
        Index dilation_m, Index dilation_n, Scalar beta, T *dst)
/*! Backward convolution of WHCN tensors to get grad of weight
 *
 * The following operation is performed:
 *      `dst` = `alpha`*`f(src1, src2)` + `beta`*`dst`,
 * where `f` operation does the following:
 *      `f[i,j,k,l]` = \sum_b \sum_m \sum_n `src1[m,n,k,b]`
 *      * `src2[(m+offset_m-dilation_m*i)/stride_m,
 *              (n+offset_n-dilation_n*j)/stride_n,l,b]`
 * with `(m + offset_m - dilation_m*i) % stride_m == 0`
 * and `(n + offset_n - dilation_n*j) % stride_n == 0`
 *
 * Generally, `src1` represents input of `Conv2d` layer, `src2` represents
 * output grad of `Conv2d` layer and `dst` represents weight grad of `Conv2d`
 * layer.
 *
 * @param[in] src1_m: Size of the first axis of `src1` array
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
 * @param[in] dilation_m: dilation effect of kernel (`dst`) array
 * @param[in] dilation_n: dilation effect of kernel (`dst`) array
 * @param[in] beta: Scalar multiplier for initial value of `dst`
 * @param[inout] dst: F-contiguous array of shape
 *      (`dst_m`, `dst_n`, `src1_channels`, `src2_channels`)
 * */
{
    using Y = typename T::repr_t;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Let `d` denote a 2-dimensional index within `dst`,
    //     `s1` denote a 2-dim index within `s1`,
    //     `o` denote a 2-dim offset
    // Then, this convolution computes
    //      `conv[d] = sum_s1 src1[s1]*src2[(s1+o-dilation*d)/stride]`
    // with a limitation that `(s1+o-dilation*d) % stride = 0`.
    // And we must satisfy condition
    //      `0 <= (s1+o-dilation*d)/stride <= src2.shape-1`
    // It means all values of `d`, that actually get non-zero `conv[d]` are:
    //      `s1+o-stride*(src2.shape-1) <= dilation*d <= s1+o`
    // Which turns into
    //      `d >= ceil((o-stride*(src2.shape-1))/dilation)`
    //      `d < floor((src1.shape-1+o+dilation)/dilation)`
    Index dst_i = i % dst_m;
    i = i / dst_m;
    Index dst_j = i % dst_n;
    i = i / dst_n;
    Index ic = i % src1_channels;
    Index oc = i / src1_channels;
    // Early exit if such destination is out of bounds
    if(oc >= src2_channels)
    {
        return;
    }
    Index dst_start_m = ::max(
            (offset_m-stride_m*(src2_m-1)+dilation_m-1) / dilation_m,
            Index(0));
    Index dst_end_m = ::min((src1_m-1+offset_m+dilation_m) / dilation_m,
            dst_m);
    Index dst_start_n = ::max(
            (offset_n-stride_n*(src2_n-1)+dilation_n-1) / dilation_n,
            Index(0));
    Index dst_end_n = ::min((src1_n-1+offset_n+dilation_n) / dilation_n,
            dst_n);
    T *dst_val = dst + ((oc*src1_channels+ic)*dst_n+dst_j)*dst_m + dst_i;
    if(dst_i >= dst_start_m and dst_i < dst_end_m and
            dst_j >= dst_start_n and dst_j < dst_end_n)
    {
        Index src1_step = src1_channels * src1_n * src1_m;
        Index src2_step = src2_channels * src2_n * src2_m;
        // Additional variables for Kahan summation rule
        Y conv{0.0}, c{0}, y, t;
        // Once again, we must satisfy conditions
        //      `0 <= s1+o-dilation*d <= stride*(src2.shape-1)`
        // while `(s1+o-dilation*d) % stride == 0`
        // Therefore, indices `s1` are bound as follows:
        //      `s1 >= dilation*d-o`
        //      `s1 < dilation*d-o+stride*(src2.shape-1)+1`
        Index src1_start_m = dilation_m*dst_i - offset_m;
        if(src1_start_m < 0)
        {
            Index neg_rem = (-src1_start_m) % stride_m;
            // We need to get minimal non-negative number
            // with the same reminder
            if(neg_rem == 0)
            {
                src1_start_m = 0;
            }
            else
            {
                src1_start_m = stride_m - neg_rem;
            }
        }
        Index src1_end_m = ::min(
                dilation_m*dst_i - offset_m + stride_m*(src2_m-1) + 1,
                src1_m);
        Index src1_start_n = dilation_n*dst_j - offset_n;
        if(src1_start_n < 0)
        {
            Index neg_rem = (-src1_start_n) % stride_n;
            // We need to get minimal non-negative number
            // with the same reminder
            if(neg_rem == 0)
            {
                src1_start_n = 0;
            }
            else
            {
                src1_start_n = stride_n - neg_rem;
            }
        }
        Index src1_end_n = ::min(
                dilation_n*dst_j - offset_n + stride_n*(src2_n-1) + 1,
                src1_n);
        for(Index src1_i = src1_start_m; src1_i < src1_end_m;
                src1_i += stride_m)
        {
            for(Index src1_j = src1_start_n;
                    src1_j < src1_end_n; src1_j += stride_n)
            {
                const T *src1_slice = src1 + src1_i
                    + (ic*src1_n+src1_j)*src1_m;
                // Slice of `src2[(s1+o-dilation*d)/stride]`
                const T *src2_slice = src2
                    + (src1_i + offset_m - dilation_m*dst_i) / stride_m
                    + (oc*src2_n +
                            (src1_j+offset_n-dilation_n*dst_j) / stride_n
                      )*src2_m;
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
void cuda(cudaStream_t stream, Index src1_m, Index src1_n, Index src1_channels,
        Index batch, Index src2_m, Index src2_n, Index stride_m,
        Index stride_n, Index src2_channels, Index offset_m, Index offset_n,
        Scalar alpha, const T *src1, const T *src2, Index dst_m, Index dst_n,
        Index dilation_m, Index dilation_n, Scalar beta, T *dst)
    noexcept
{
    int nelems_dst = dst_m * dst_n * src1_channels * src2_channels;
    dim3 blocks((nelems_dst+255)/256), threads(256);
    cuda_kernel<T><<<blocks, threads, 0, stream>>>(src1_m, src1_n,
            src1_channels, batch, src2_m, src2_n, stride_m, stride_n,
            src2_channels, offset_m, offset_n, alpha, src1, src2, dst_m, dst_n,
            dilation_m, dilation_n, beta, dst);
}

// Explicit instantiation
template
void cuda<bf16_t>(cudaStream_t stream, Index src1_m, Index src1_n,
        Index src1_channels, Index batch, Index src2_m, Index src2_n,
        Index stride_m, Index stride_n, Index src2_channels, Index offset_m,
        Index offset_n, Scalar alpha, const bf16_t *src1, const bf16_t *src2,
        Index dst_m, Index dst_n, Index dilation_m, Index dilation_n,
        Scalar beta, bf16_t *dst)
    noexcept;

template
void cuda<fp32_t>(cudaStream_t stream, Index src1_m, Index src1_n,
        Index src1_channels, Index batch, Index src2_m, Index src2_n,
        Index stride_m, Index stride_n, Index src2_channels, Index offset_m,
        Index offset_n, Scalar alpha, const fp32_t *src1, const fp32_t *src2,
        Index dst_m, Index dst_n, Index dilation_m, Index dilation_n,
        Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index src1_m, Index src1_n,
        Index src1_channels, Index batch, Index src2_m, Index src2_n,
        Index stride_m, Index stride_n, Index dst_channels, Index offset_m,
        Index offset_n, Scalar alpha, const fp32_fast_tf32_t *src1,
        const fp32_fast_tf32_t *src2, Index dst_m, Index dst_n,
        Index dilation_m, Index dilation_n, Scalar beta, fp32_fast_tf32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index src1_m, Index src1_n,
        Index src1_channels, Index batch, Index src2_m, Index src2_n,
        Index stride_m, Index stride_n, Index src2_channels, Index offset_m,
        Index offset_n, Scalar alpha, const fp64_t *src1, const fp64_t *src2,
        Index dst_m, Index dst_n, Index dilation_m, Index dilation_n,
        Scalar beta, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::conv2d_bwd_weight_inplace
