/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/maxsumexp/cpu.cc
 * Max and sum of exponents of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-07
 * */

#include "nntile/kernel/maxsumexp/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace maxsumexp
{

template<typename T>
void cpu(Index m, Index n, Index k, const T *src, T *maxsumexp)
    noexcept
//! Max and sum of exponents along middle axis
/*! For a provided m-by-k-by-n input array src compute maximums and sums of
 * exponents of slices along second axis with k elements, resulting in
 * 2-by-m-by-n output array maxsumexp.
 *
 * Mnemonically, the following operations are performed:
 *      old[0,i,j] = maxsumexp[0,i,j]
 *      old[1,i,j] = maxsumexp[1,i,j]
 *      maxsumexp[0,i,j] = max(old[0,i,j], max(src[i,:,j]))
 *      maxsumexp[1,i,j] = old[1,i,j]*exp(old[0,i,j]-maxsumexp[0,i,j])
 *          + sum(exp(src[i,:,j]-maxsumexp[0,i,j])))
 *
 * @param[in] m: Size of the first mode of src and the second mode of sumnorm
 *      arrays.
 * @param[in] n: Size of the last mode of src and sumnorm arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[inout] maxsumexp: Output contiguous 2-by-m-by-n array, that accumulates
 *      maximums and sums of exponents of slices along middle axis.
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
            // Get max and sum of exponents of a corresponding slice
            const T *src_slice = src + i2*mk + i1;
            // Init max and sum
            T max = maxsumexp[dst_offset];
            T sum = maxsumexp[dst_offset+1];
            // Check if sum is zero, which means values were not yet
            // initialized. Just initialize maximum value in this case.
            if(sum == zero)
            {
                max = src_slice[0];
            }
            // Cycle over slice of input buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_slice[i0*m];
                // Update max and sum of exponents
                if(max < val)
                {
                    sum = sum*std::exp(max-val) + one;
                    max = val;
                }
                else
                {
                    sum += std::exp(val-max);
                }
            }
            // Save result
            maxsumexp[dst_offset] = max;
            maxsumexp[dst_offset+1] = sum;
            dst_offset += 2;
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, const fp32_t *src,
        fp32_t *maxsumexp)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, const fp64_t *src,
        fp64_t *maxsumexp)
    noexcept;

} // namespace maxsumexp
} // namespace kernel
} // namespace nntile

