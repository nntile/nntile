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
void cpu(Index m, Index n, Index k, const T *src, Scalar beta, T *maxsumexp)
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
 * beta=0 overwrites maxsumexp (old ignored). beta=1 accumulates.
 *
 * @param[in] m: Size of the first mode of src and the second mode of sumexp
 *      arrays.
 * @param[in] n: Size of the last mode of src and sumexp arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: 0.0 overwrite, 1.0 accumulate
 * @param[inout] maxsumexp: Output contiguous 2-by-m-by-n array
 * */
{
    using Y = typename T::repr_t;
    const Index mk = m * k;
    Index dst_offset = 0;
    constexpr Y zero{0.0}, one{1.0};
    const bool overwrite = (beta == 0.0);
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
            // Save result
            if(not std::isinf(max))
            {
                if(overwrite)
                {
                    maxsumexp[dst_offset] = static_cast<T>(max);
                    maxsumexp[dst_offset+1] = static_cast<T>(sum);
                }
                else
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
                            y = sum_old*std::exp(max_old-max) - c;
                            maxsumexp[dst_offset+1] = static_cast<T>(sum + y);
                        }
                        else
                        {
                            Y tmp = std::exp(max-max_old);
                            y = sum_old - c*tmp;
                            sum *= tmp;
                            maxsumexp[dst_offset+1] = static_cast<T>(sum + y);
                        }
                    }
                }
            }
            else if(overwrite)
            {
                // All-masked fiber: write zeros so STARPU_W leaves no garbage
                maxsumexp[dst_offset] = static_cast<T>(zero);
                maxsumexp[dst_offset+1] = static_cast<T>(zero);
            }
            dst_offset += 2;
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, const fp32_t *src, Scalar beta,
        fp32_t *maxsumexp)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, const fp64_t *src, Scalar beta,
        fp64_t *maxsumexp)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, const bf16_t *src, Scalar beta,
        bf16_t *maxsumexp)
    noexcept;

template
void cpu<fp16_t>(Index m, Index n, Index k, const fp16_t *src, Scalar beta,
        fp16_t *maxsumexp)
    noexcept;

} // namespace nntile::kernel::maxsumexp
