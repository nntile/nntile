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
 * @author K. Sozykin
 * @date 2023-02-27
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
void cuda_kernel(Index m, Index n, Index k, Index mk, const T *src,
        T *sum_dst)
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    constexpr T zero = 0, one = 1;
    // Cycle over row of output buffer
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        // Cycle over column of output buffer
        for(Index i1 = i1_start; i1 < m; i1 += i1_step)
        {
            // Get sum of a corresponding slice
            const T *src_slice = src + i2*mk + i1;
            Index dst_offset = (i1+i2*m);
            // Init sum 
            T sum = sum_dst[dst_offset];
            // Cycle over slice of input buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_slice[i0*m];
                // Update sum
                sum += val;
                
            }
            // Save result
            sum_dst[dst_offset] = sum;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *src,
        T *sum_dst)
    noexcept
//! Sum and Euclidian norm along middle axis
/*! For a provided m-by-k-by-n input array src compute sums  of slices
 * along second axis with k elements, resulting in m-by-n output array
 * sum. Input value sum[i, j] is increased by a sum of elements of a
 * slice src[i, :, j] on output. Values of array sum are updated by this routine in
 * read-write mode, therefore sumnorm must be initialized before use with zeros
 * (e.g., by clear() function).
 *
 * Mnemonically, the following operations are performed:
 *      sum[i,j] = sum[i,j] + sum(src[i,:,j])
 *      
 *
 * @param[in] m: Size of the first mode of src and the second mode of sumnorm
 *      arrays.
 * @param[in] n: Size of the last mode of src and sumnorm arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[inout] sum: Output contiguous m-by-n array, that accumulates
 *      sums along middle axis.
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    dim3 blocks(16, 16), threads(8, 4);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, src,
            sum_dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *src, fp32_t *sum_dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *src, fp64_t *sum_dst)
    noexcept;

} // namespace sum
} // namespace kernel
} // namespace nntile

