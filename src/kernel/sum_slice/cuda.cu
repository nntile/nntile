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
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    constexpr T zero = 0;
    if(i0 < m and i1 < n)
    {
        // Pointer to a corresponding fiber of the source array src
        const T *src_fiber = src + i1*mk + i0;
        // Init sum over the fiber
        T sum = zero;
        // Output value
        T &result = dst[i1*m+i0];
        // Cycle over fiber elements and accumulate the sum
        for(Index i2 = 0; i2 < k; ++i2)
        {
            sum += src_fiber[i2*m];
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
    dim3 threads(std::min(int(m), 16), std::min(int(n), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y);
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

