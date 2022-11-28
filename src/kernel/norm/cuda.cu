/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm/cuda.cu
 * Euclidian norm of a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-28
 * */

#include "nntile/kernel/norm/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace norm
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, const T *src, T *norm)
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
            // Get norm of a corresponding slice
            const T *src_slice = src + i2*mk + i1;
            Index dst_offset = i1 + i2*m;
            // Norm is computed with help of scaled sum of squares
            T scale = norm[dst_offset];
            T ssq = one;
            // Cycle over slice of input buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_slice[i0*m];
                // Nothing to update in case of 0
                if(val == zero)
                {
                    continue;
                }
                // Update scale and scaled sum of squares
                T absval = abs(val);
                if(absval > scale)
                {
                    T tmp = scale / absval;
                    scale = absval;
                    ssq = ssq*tmp*tmp + one;
                }
                else
                {
                    T tmp = absval / scale;
                    ssq += tmp*tmp;
                }
            }
            // Save result
            norm[dst_offset] = scale * sqrt(ssq);
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *src,
        T *norm)
    noexcept
//! Euclidian norm along middle axis
/*! For a provided m-by-k-by-n input array src compute norms of slices
 * along second axis with k elements, resulting in by-m-by-n output array
 * norm. Output value of norm[i, j] is a
 * square root of sum of squares of input norm[i, j] and norm of a slice
 * src[i, :, j]. Values of array norm are updated by this routine in
 * read-write mode, therefore norm must be initialized before use with zeros
 * (e.g., by clear() function).
 *
 * Mnemonically, the following operations are performed:
 *      norm[i,j] = sqrt(norm[i,j] + norm(src[i,:,j])^2)
 *
 * @param[in] m: Size of the first mode of src and norm arrays
 * @param[in] n: Size of the last mode of src and norm arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[inout] norm: Output contiguous m-by-n array, that accumulates
 *      norms of slices along middle axis.
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    dim3 blocks(16, 16), threads(8, 4);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, src,
            norm);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *src, fp32_t *norm)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *src, fp64_t *norm)
    noexcept;

} // namespace norm
} // namespace kernel
} // namespace nntile

