/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sumprod_fiber/cuda.cu
 * Sums over slices into a fiber of a product of buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sumprod_fiber/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::sumprod_fiber
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Scalar alpha_, const T *src1,
        const T *src2, Scalar beta_, T *dst)
//! Sums over slices into a fiber of a product of two tensors on CPU
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding slices along the first and the third
 * axes with m and n elements respectively, resulting in output vector dst with
 * k elements.
 * Mnemonically, the following operations are performed:
 *      dst[l] = beta*dst[l] + alpha*sum_ij(src1[i,l,j] * src2[i,l,j])
 *
 * @param[in] m: Size of the first mode of src1 and src2 tensors
 * @param[in] n: Size of the last mode of src1 and src2 tensors
 * @param[in] k: Size of the middle mode of src1 and src2 tensors and of the
 *      only mode of dst tensor
 * @param[in] alpha_: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that
 *      accumulates sums along the first and the last axes of per-element
 *      products of src1 and src2.
 * */
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x;
    Index i0_start = threadIdx.y, i0_step = blockDim.y;
    Index i1_start = threadIdx.z, i1_step = blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    constexpr Y zero{0.0};
    // Init sum of product of the slices
    Y sum = zero;
    if(i2 < k)
    {
        // Cycle over column of src1 and src2
        for(Index i1 = i1_start; i1 < n; i1 += i1_step)
        {
            // Get corresponding fibers of both sources
            const T *src1_fiber = src1 + (i1*k+i2)*m;
            const T *src2_fiber = src2 + (i1*k+i2)*m;
            // Cycle over fibers of inputs
            for(Index i0 = i0_start; i0 < m; i0 += i0_step)
            {
                // Update sum
                sum += Y{src1_fiber[i0]} * Y{src2_fiber[i0]};
            }
        }
    }
    __shared__ Y block_sum[2];
    if(i1_start == 0 and i0_start == 0)
    {
        block_sum[threadIdx.x] = zero;
    }
    __syncthreads();
    atomicAdd(&block_sum[threadIdx.x], sum);
    __syncthreads();
    // Update output value
    if(i1_start == 0 and i0_start == 0 and i2 < k)
    {
        // Output value
        T &result = dst[i2];
        if(beta == zero)
        {
            result = T{alpha * Y{block_sum[threadIdx.x]}};
        }
        else
        {
            result = T{beta * Y{result} + alpha * block_sum[threadIdx.x]};
        }
    }
}

template<typename T, int BLOCK_ROW, int BLOCK_COL, int LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src1,
        const T *src2, Scalar beta_, T *dst)
//! Sums over slices into a fiber of a product of two tensors on CPU
/*! For two provided 1-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding slices along the first and the third
 * axes with m and n elements respectively, resulting in output vector dst with
 * k elements.
 * Mnemonically, the following operations are performed:
 *      dst[l] = beta*dst[l] + alpha*sum_j(src1[0,l,j] * src2[0,l,j])
 *
 * @param[in] n: Size of the last mode of src1 and src2 tensors
 * @param[in] k: Size of the middle mode of src1 and src2 tensors and of the
 *      only mode of dst tensor
 * @param[in] alpha_: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous 1-by-k-by-n array
 * @param[in] src2: Input contiguous 1-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that
 *      accumulates sums along the first and the last axes of per-element
 *      products of src1 and src2.
 * */
{
    Index src_block_j_end = (n/BLOCK_COL) * BLOCK_COL;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    constexpr int BLOCK_COL_STEP = BLOCK_COL / LOOP;
    __shared__ Y dst_block[BLOCK_ROW][BLOCK_COL_STEP];
    Y dst_val = 0.0;
    Index src_l = threadIdx.x % BLOCK_ROW;
    Index src_j = threadIdx.x / BLOCK_ROW;
    Index src_offset = blockIdx.x*BLOCK_ROW + src_l + src_j*k;
    // Pointer to a corresponding fiber of the input arrays
    if(src_l+blockIdx.x*BLOCK_ROW < k)
    {
        for(Index src_block_j = 0; src_block_j < src_block_j_end;
                src_block_j += BLOCK_COL)
        {
            const T *src1_fiber = src1 + src_offset + src_block_j*k;
            const T *src2_fiber = src2 + src_offset + src_block_j*k;
            for(int c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                Y val1 = static_cast<Y>(src1_fiber[c*k]);
                Y val2 = static_cast<Y>(src2_fiber[c*k]);
                dst_val += val1 * val2;
            }
        }
        // Pointer to a corresponding fiber of the input arrays
        const T *src1_fiber = src1 + src_offset + src_block_j_end*k;
        const T *src2_fiber = src2 + src_offset + src_block_j_end*k;
        for(Index c = 0; c < n-src_block_j_end; c += BLOCK_COL_STEP)
        {
            Y val1 = static_cast<Y>(src1_fiber[c*k]);
            Y val2 = static_cast<Y>(src2_fiber[c*k]);
            dst_val += val1 * val2;
        }
    }
    // Put calculated value into shared memory
    dst_block[src_l][src_j] = alpha * dst_val;
    // Inter-warp reduction
    for(int c = BLOCK_COL_STEP>>1; c > 0; c >>= 1)
    {
        __syncthreads();
        if(src_j < c)
        {
            dst_block[src_l][src_j] += dst_block[src_l][src_j+c];
        }
    }
    // Write output
    if(src_j == 0 and src_l+blockIdx.x*BLOCK_ROW < k)
    {
        if(beta == 0.0)
        {
            dst[blockIdx.x*BLOCK_ROW+src_l] = static_cast<T>(
                    static_cast<Y>(dst_block[src_l][0]));
        }
        else
        {
            dst_val = beta * static_cast<Y>(dst[blockIdx.x*BLOCK_ROW+src_l]);
            dst[blockIdx.x*BLOCK_ROW+src_l] = static_cast<T>(
                    dst_val + dst_block[src_l][0]);
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src1, const T *src2, Scalar beta, T *dst)
    noexcept
//! Sums over slices into a fiber of a product of two tensors on CPU
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding slices along the first and the third
 * axes with m and n elements respectively, resulting in output vector dst with
 * k elements.
 * Mnemonically, the following operations are performed:
 *      dst[l] = beta*dst[l] + alpha*sum_ij(src1[i,l,j] * src2[i,l,j])
 *
 * @param[in] m: Size of the first mode of src1 and src2 tensors
 * @param[in] n: Size of the last mode of src1 and src2 tensors
 * @param[in] k: Size of the middle mode of src1 and src2 tensors and of the
 *      only mode of dst tensor
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that
 *      accumulates sums along the first and the last axes of per-element
 *      products of src1 and src2.
 * */
{
    // Both source and destination are Fortran-contiguous
    // Separate case for m==1
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks((k+31)/32);
        (cuda_kernel_m1<T, 32, 8, 1>)<<<blocks, threads, 0, stream>>>(n, k,
                alpha, src1, src2, beta, dst);
    }
    else
    {
        dim3 threads(2, std::min(int(m), 32), std::min(int(n), 32));
        dim3 blocks((k+threads.x-1)/threads.x, 1, 1);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, alpha, src1,
                src2, beta, dst);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src1, const fp32_t *src2, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src1, const fp64_t *src2, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src1, const bf16_t *src2, Scalar beta, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src1, const fp16_t *src2, Scalar beta, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::sumprod_fiber
