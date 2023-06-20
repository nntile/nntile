/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sum_fiber/cuda.cu
 * Sums over slices into a fiber of a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-06-20
 * */

#include "nntile/kernel/sum_fiber/cuda.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace sum_fiber
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, T alpha, const T *src, T beta,
        T *dst)
//! Sums over slices along the first and last axes into a fiber of a tensor
/*! For a provided m-by-k-by-n input array computes sums over slices
 * along the first axis with m elements and the last axis with n elements,
 * resulting in output fiber of shape (k).
 * Mnemonically, the following operations are performed:
 *      dst[k] = beta*dst[k] + alpha*sum(src[:,k,:])
 *
 * @param[in] m: Size of the first mode of src array
 * @param[in] n: Size of the last mode of src array
 * @param[in] k: Size of the middle mode of src array and the only mode of
 *      dst array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] sum: Output contiguous vector with k elements, that accumulate
 *      sums over slices along the first and the last axes.
 * */
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i2_step = blockDim.x * gridDim.x;
    constexpr T zero = 0;
    // Cycle over the only axis of output buffer
    for(Index i2 = i2_start; i2 < k; i2 += i2_step)
    {
        // Init sum 
        T sum = zero;
        // Cycle over the third axis of input buffer
        for(Index i1 = 0; i1 < n; ++i1)
        {
            // Get sum of a corresponding slice
            const T *src_slice = src + (i1*k+i2)*m;
            // Cycle over the first axis of input buffer
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Read value from source
                T val = src_slice[i0];
                // Update sum
                sum += val;
            }
        }
        // Save result
        if(beta == zero)
        {
            sum *= alpha;
        }
        else
        {
            sum = beta*dst[i2] + alpha*sum;
        }
        dst[i2] = sum;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src, T beta, T *dst)
    noexcept
//! Sums over slices along the first and last axes into a fiber of a tensor
/*! For a provided m-by-k-by-n input array computes sums over slices
 * along the first axis with m elements and the last axis with n elements,
 * resulting in output fiber of shape (k).
 * Mnemonically, the following operations are performed:
 *      dst[k] = beta*dst[k] + alpha*sum(src[:,k,:])
 *
 * @param[in] m: Size of the first mode of src array
 * @param[in] n: Size of the last mode of src array
 * @param[in] k: Size of the middle mode of src array and the only mode of
 *      dst array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] sum: Output contiguous vector with k elements, that accumulate
 *      sums over slices along the first and the last axes.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 blocks(256), threads(32);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, alpha, src, beta,
            dst);
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

} // namespace sum_fiber
} // namespace kernel
} // namespace nntile

