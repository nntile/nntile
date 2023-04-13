/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sum/cuda.cc
 * Sum of a buffer on GPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Konstantin Sozykin
 * @date 2023-04-13
 * */


#include "nntile/kernel/sum/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace sum
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, T alpha, const T *src,
        T beta, T *sum_dst)
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    constexpr T zero = 0;
    // Cycle over row of output buffer
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        // Cycle over column of output buffer
        for(Index i1 = i1_start; i1 < m; i1 += i1_step)
        {
            // Get sum of a corresponding slice
            const T *src_slice = src + i2*mk + i1;
            Index dst_offset = i1 + i2*m;
            // Init sum
            T sum = zero;
            // Cycle over slice of input buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_slice[i0*m];
                // Update sum
                sum += val;
            }
            // Save result
            if(beta == zero)
            {
                sum *= alpha;
            }
            else
            {
                sum = beta*sum_dst[dst_offset] + alpha*sum;
            }
            sum_dst[dst_offset] = sum;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src, T beta, T *sum_dst)
    noexcept
//! Sum along middle axis
/*! For a provided m-by-k-by-n input array src compute sums of slices
 * along second axis with k elements, resulting in m-by-n output array
 * sum_dst. Mnemonically, the following operations are performed:
 *      sum[i,j] = beta*sum[i,j] + alpha*sum(src[i,:,j])
 *
 * @param[in] m: Size of the first mode of src and sum_dst arrays
 * @param[in] n: Size of the last mode of src and sum_dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for sum_dst
 * @param[inout] sum: Output contiguous m-by-n array, that accumulates
 *      sums along middle axis.
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    dim3 blocks(16, 16), threads(8, 4);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha, src,
            beta, sum_dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, fp32_t alpha,
        const fp32_t *src, fp32_t beta, fp32_t *sum_dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, fp64_t alpha,
        const fp64_t *src, fp64_t beta, fp64_t *sum_dst)
    noexcept;

} // namespace sum
} // namespace kernel
} // namespace nntile

