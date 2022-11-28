/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm/cpu.cc
 * Euclidian norm of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-28
 * */

#include "nntile/kernel/norm/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace norm
{

template<typename T>
void cpu(Index m, Index n, Index k, const T *src, T *norm)
    noexcept
//! Euclidian norm along middle axis
/*! For a provided m-by-k-by-n input array src compute norms of slices
 * along second axis with k elements, resulting in m-by-n output array
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
    const Index mk = m * k;
    Index dst_offset = 0;
    constexpr T zero = 0, one = 1;
    // Cycle over row of output buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over column of output buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Get norm of a corresponding slice
            const T *src_slice = src + i2*mk + i1;
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
                T absval = std::abs(val);
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
            norm[dst_offset] = scale * std::sqrt(ssq);
            ++dst_offset;
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, const fp32_t *src, fp32_t *norm)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, const fp64_t *src, fp64_t *norm)
    noexcept;

} // namespace norm
} // namespace kernel
} // namespace nntile

