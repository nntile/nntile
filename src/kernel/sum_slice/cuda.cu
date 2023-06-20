/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sum_slice/cuda.cu
 * Sums over fibers into a slice of a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Konstantin Sozykin
 * @date 2023-06-20
 * */

#include "nntile/kernel/sum_slice/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace sum_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, T alpha, const T *src,
        T beta, T *dst)
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    constexpr T zero = 0;
    // Cycle over column of output buffer
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        // Cycle over row of output buffer
        for(Index i1 = i1_start; i1 < m; i1 += i1_step)
        {
            // Pointer to a corresponding fiber of the source array src
            const T *src_fiber = src + i2*mk + i1;
            // Init sum over the fiber
            T sum = zero;
            // Output value
            T &result = dst[i2*m+i1];
            // Cycle over fiber elements and accumulate the sum
            for(Index i0 = 0; i0 < k; ++i0)
            {
                sum += src_fiber[i0*m];
            }
            // Update output value
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
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src, T beta, T *dst)
    noexcept
//! Sums over fibers along middle axis into a slice of a tensor
/*! For a provided m-by-k-by-n input array computes sums over fibers
 * along second axis with k elements, resulting in m-by-n output slice.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*sum(src[i,:,j])
 *
 * @param[in] m: Size of the first mode of src and dst arrays
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] sum: Output contiguous m-by-n array, that accumulates
 *      sums over fibers along middle axis.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 blocks(16, 16), threads(8, 4);
    if(m == 1)
    {
        blocks = 256;
        threads = 32;
    }
    else if(n == 1)
    {
        blocks = dim3(1, 256);
        threads = dim3(1, 32);
    }
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha, src,
            beta, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, fp32_t alpha,
        const fp32_t *src, fp32_t beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, fp64_t alpha,
        const fp64_t *src, fp64_t beta, fp64_t *dst)
    noexcept;

} // namespace sum_slice
} // namespace kernel
} // namespace nntile

