/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sum/cpu.cc
 * Sum of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Konstantin Sozykin
 * @date 2023-02-19
 * */

#include "nntile/kernel/sum/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace sum
{

template<typename T>
void cpu(Index m, Index n, Index k, const T *src, T *sum_dst)
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
    const Index mk = m * k;
    Index dst_offset = 0;
    constexpr T zero = 0;
    // Cycle over row of output buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over column of output buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Get sum and norm of a corresponding slice
            const T *src_slice = src + i2*mk + i1;
            // Init sum 
            // Norm is computed with help of scaled sum of squares
            T sum = sum_dst[dst_offset];
            // Cycle over slice of input buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_slice[i0*m];
                sum += val;
            }
            // Save result. 
            sum_dst[dst_offset] = sum;
            dst_offset += 1;
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, const fp32_t *src,
        fp32_t *sum_dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, const fp64_t *src,
        fp64_t *sum_dst)
    noexcept;

} // namespace sum
} // namespace kernel
} // namespace nntile

