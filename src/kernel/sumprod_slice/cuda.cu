/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sumprod_slice/cuda.cu
 * Sums over fibers into a slice of a product of buffers on CUDA
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/sumprod_slice/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::sumprod_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, T alpha, const T *src1,
        const T *src2, T beta, T *dst)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    constexpr T zero = 0;
    if(i0 < m and i1 < n)
    {
        // Get corresponding fibers of both sources
        const T *src1_fiber = src1 + i1*mk + i0;
        const T *src2_fiber = src2 + i1*mk + i0;
        // Init sum of product of the fibers
        T sum = zero;
        // Cycle over fibers of inputs
        for(Index i2 = i2_start; i2 < k; i2 += i2_step)
        {
            // Update sum
            sum += src1_fiber[i2*m] * src2_fiber[i2*m];
        }
        __shared__ T block_sum[64];
        if(i2_start == 0)
        {
            block_sum[threadIdx.x+blockDim.x*threadIdx.y] = zero;
        }
        __syncthreads();
        atomicAdd(&block_sum[threadIdx.x+blockDim.x*threadIdx.y], sum);
        __syncthreads();
        // Update output value
        if(i2_start == 0)
        {
            // Output value
            T &result = dst[i1*m+i0];
            sum = block_sum[threadIdx.x+blockDim.x*threadIdx.y];
            if(beta == zero)
            {
                result = alpha * sum;
            }
            else
            {
                result = beta*result + alpha*sum;
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, scal_t alpha,
        const T *src1_, const T *src2_, scal_t beta, T *dst_)
    noexcept
//! Sums over fibers into a slice of a product of two tensors
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding fibers along second axis with k
 * elements, resulting in m-by-n output array dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*sum_l(src1[i,l,j] * src2[i,l,j])
 *
 * @param[in] m: Size of the first mode of src1, src2 and dst
 * @param[in] n: Size of the last mode of src1, src2 and dst
 * @param[in] k: Size of the middle mode of src1 and src2 arrays
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1_: Input contiguous m-by-k-by-n array
 * @param[in] src2_: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst_: Output contiguous m-by-n array, that accumulates
 *      sums along middle axis of per-element products of src1 and src2.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8), 16);
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
    using Y = typename CUDAComputeType<T>::value;
    auto src1 = reinterpret_cast<const Y *>(src1_);
    auto src2 = reinterpret_cast<const Y *>(src2_);
    auto dst = reinterpret_cast<Y *>(dst_);
    (cuda_kernel<Y>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, Y{alpha},
            src1, src2, Y{beta}, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, scal_t alpha,
        const fp32_t *src1, const fp32_t *src2, scal_t beta, fp32_t *sum_dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, scal_t alpha,
        const fp64_t *src1, const fp64_t *src2, scal_t beta, fp64_t *sum_dst)
    noexcept;

} // namespace nntile::kernel::sumprod_slice
