/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm_slice/cuda.cu
 * Euclidean norms of fibers into a slice of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/norm_slice/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::norm_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_, const T *src,
        Scalar beta_, T *dst)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    using Y = typename T::repr_t;
    const Y beta{beta_};
    const Y alpha{alpha_};
    using Z = typename CUDAComputeType<T>::value;
    constexpr Y zero{0.};
    if(i0 < m and i1 < n)
    {
        // Pointer to a corresponding fiber of the source array src
        const T *src_fiber = src + i1*mk + i0;
        // Init sum over the fiber
        Y sum = zero;
        // Cycle over fiber elements and accumulate the sum
        for(Index i2 = i2_start; i2 < k; i2 += i2_step)
        {
            sum = ::hypot(sum, Y{src_fiber[i2*m]});
        }
        volatile __shared__ Y block_max[64];
        __shared__ Y block_sum[64];
        if(i2_start == 0)
        {
            block_max[threadIdx.x+blockDim.x*threadIdx.y] = sum;
            block_sum[threadIdx.x+blockDim.x*threadIdx.y] = zero;
        }
        __syncthreads();
        while(block_max[threadIdx.x+blockDim.x*threadIdx.y] < sum)
        {
            block_max[threadIdx.x+blockDim.x*threadIdx.y] = sum;
        }
        __syncthreads();
        if(block_max[threadIdx.x+blockDim.x*threadIdx.y] > 0)
        {
            sum /= block_max[threadIdx.x+blockDim.x*threadIdx.y];
            atomicAdd(&block_sum[threadIdx.x+blockDim.x*threadIdx.y], sum*sum);
            __syncthreads();
            // Update output value
            if(i2_start == 0)
            {
                // Output value
                T &result = dst[i1*m+i0];
                sum = block_max[threadIdx.x+blockDim.x*threadIdx.y];
                sum *= ::sqrt(block_sum[threadIdx.x+blockDim.x*threadIdx.y]);
                if(beta == zero)
                {
                    result = T{::fabs(alpha) * sum};
                }
                else
                {
                    result = T{::hypot(beta*Y{result}, alpha*sum)};
                }
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src, Scalar beta, T *dst)
    noexcept
//! Euclidean norms over fibers along middle axis into a slice of a tensor
/*! For a provided m-by-k-by-n input array src compute norms of fibers
 * along second axis with k elements, resulting in m-by-n output array-slice
 * dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = hypot(beta*dst[i,j], alpha*norm(src[i,:,j]))
 *
 * @param[in] m: Size of the first mode of src and dst arrays
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src_: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst_: Input and output contiguous m-by-n array, that
 *      accumulates norms along middle axis.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
            src, beta, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::norm_slice
