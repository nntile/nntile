/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/maxsumexp/cpu.cc
 * Max and sum of exponents of a buffer on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/maxsumexp/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::maxsumexp
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
 * @param[inout] maxsumexp: Output contiguous 2-by-m-by-n array, that
 *      accumulates maximums and sums of exponents of slices along middle axis.
 * */
{
    using Y = typename T::repr_t;
    const Index mk = m * k;
    Index dst_offset = 0;
    constexpr Y zero{0.0}, one{1.0};
    // Cycle over row of output buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over column of output buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Get max and sum of exponents of a corresponding slice
            const T *src_slice = src + i2*mk + i1;
            // Init max and sum with the first value
            Y max = static_cast<Y>(src_slice[0]);
            Y sum{one}, c{zero}, y, t;
            // Cycle over slice of input buffer
            for(Index i0 = 1; i0 < k; ++i0)
            {
                // Read value from source
                Y val = static_cast<Y>(src_slice[i0*m]);
                // Ignore -inf value, which comes from mask
                if(std::isinf(val))
                {
                    continue;
                }
                // Update max and sum of exponents
                if(max < val)
                {
                    //sum = sum*std::exp(max-val) + one;
                    Y tmp = std::exp(max-val);
                    y = one - c*tmp;
                    sum *= tmp;
                    t = sum + y;
                    c = (t-sum) - y;
                    sum = t;
                    max = val;
                }
                else
                {
                    //sum += std::exp(val-max);
                    y = std::exp(val-max) - c;
                    t = sum + y;
                    c = (t-sum) - y;
                    sum = t;
                }
            }
            // Save result, do nothing if all elements are masked out
            if(not std::isinf(max))
            {
                Y sum_old = static_cast<Y>(maxsumexp[dst_offset+1]);
                // If old sum is zero then just overwrite it with current sum
                if(sum_old == zero)
                {
                    maxsumexp[dst_offset] = static_cast<T>(max);
                    maxsumexp[dst_offset+1] = static_cast<T>(sum);
                }
                // Update non-zero initial sum
                else
                {
                    Y max_old = static_cast<Y>(maxsumexp[dst_offset]);
                    if(max_old < max)
                    {
                        maxsumexp[dst_offset] = static_cast<T>(max);
                        maxsumexp[dst_offset+1] = static_cast<T>(sum_old*std::exp(max_old-max)
                            + sum);
                        y = sum_old*std::exp(max_old-max) - c;
                        maxsumexp[dst_offset+1] = static_cast<T>(sum + y);
                    }
                    else
                    {
                        maxsumexp[dst_offset+1] = static_cast<T>(sum*std::exp(max-max_old)
                            + sum_old);
                        Y tmp = std::exp(max-max_old);
                        y = sum_old - c*tmp;
                        sum *= tmp;
                        maxsumexp[dst_offset+1] = static_cast<T>(sum + y);
                    }
                }
            }
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
void cpu<fp32_fast_tf32_t>(Index m, Index n, Index k, const fp32_fast_tf32_t *src,
        fp32_fast_tf32_t *maxsumexp)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, const fp64_t *src,
        fp64_t *maxsumexp)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, const bf16_t *src,
        bf16_t *maxsumexp)
    noexcept;

} // namespace nntile::kernel::maxsumexp
