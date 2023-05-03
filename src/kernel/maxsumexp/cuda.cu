/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/maxsumexp/cuda.cu
 * Max and sum of exponents of a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-03
 * */

#include "nntile/kernel/maxsumexp/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace maxsumexp
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, const T *src,
        T *maxsumexp)
{
    Index i1 = threadIdx.x + blockIdx.x*blockDim.x,
          i2 = threadIdx.y + blockIdx.y*blockDim.y;
    constexpr T zero = 0, one = 1;
    // Check column index of output buffer
    if(i2 < n)
    {
        // Check row index of output buffer
        if(i1 < m)
        {
            // Get max and sum of exponents of a corresponding fiber
            const T *src_fiber = src + i2*mk + i1;
            // Init max and sum
            Index dst_offset = 2 * (i1+i2*m);
            T max = maxsumexp[dst_offset];
            T sum = maxsumexp[dst_offset+1];
            // Check if sum is zero, which means values were not yet
            // initialized. Just initialize maximum value in this case.
            if(sum == zero)
            {
                max = src_fiber[0];
            }
            // Cycle over fiber of input buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_fiber[i0*m];
                // Update max and sum of exponents
                if(max < val)
                {
                    sum = sum*exp(max-val) + one;
                    max = val;
                }
                else
                {
                    sum += exp(val-max);
                }
            }
            // Save result
            maxsumexp[dst_offset] = max;
            maxsumexp[dst_offset+1] = sum;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *src,
        T *maxsumexp)
    noexcept
//! Max and sum of exponents along middle axis
/*! For a provided m-by-k-by-n input array src compute maximums and sums of
 * exponents of slices along second axis with k elements, resulting in
 * 2-by-m-by-n output array maxsumexp.
 *
 *      old[0,i,j] = maxsumexp[0,i,j]
 *      old[1,i,j] = maxsumexp[1,i,j]
 *      maxsumexp[0,i,j] = max(old[0,i,j], max(src[i,:,j]))
 *      maxsumexp[1,i,j] = old[1,i,j]*exp(old[0,i,j]-maxsumexp[0,i,j])
 *          + sum(exp(src[i,:,j]-maxsumexp[0,i,j])))
 *
 * @param[in] m: Size of the first mode of src and the second mode of maxsumexp
 *      arrays.
 * @param[in] n: Size of the last mode of src and maxsumexp arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[inout] maxsumexp: Output contiguous 2-by-m-by-n array, that accumulates
 *      sums and norms of slices along middle axis.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 blocks((m+15)/16, (n+15)/16), threads(16, 16);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, src,
            maxsumexp);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *src, fp32_t *maxsumexp)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *src, fp64_t *maxsumexp)
    noexcept;

} // namespace maxsumexp
} // namespace kernel
} // namespace nntile

