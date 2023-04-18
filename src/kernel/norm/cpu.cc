/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm/cpu.cc
 * Norm of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
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
void cpu(Index m, Index n, Index k, T alpha, const T *src, T beta, T *norm_dst)
    noexcept
//! Norm along middle axis
/*! For a provided m-by-k-by-n input array src compute norms of slices
 * along second axis with k elements, resulting in m-by-n output array
 * norm_dst. Mnemonically, the following operations are performed:
 *      norm_dst[i,j] = hypot(beta*norm_dst[i,j], alpha*norm(src[i,:,j]))
 *
 * @param[in] m: Size of the first mode of src and norm_dst arrays
 * @param[in] n: Size of the last mode of src and norm_dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for norm_dst
 * @param[inout] norm_dst: Output contiguous m-by-n array, that accumulates
 *      norms along middle axis.
 * */
{
    const Index mk = m * k;
    Index dst_offset = 0;
    constexpr T zero = 0.0, one = 1.0;
    alpha = std::abs(alpha);
    // Cycle over row of output buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over column of output buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Get norm of a corresponding slice
            const T *src_slice = src + i2*mk + i1;
            // Init norm 
            T norm_max = zero, norm_ssq = zero;
            // Cycle over slice of input buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = std::abs(src_slice[i0*m]);
                // Update norm only if new value is non-zero
                if(val > 0)
                {
                    if(norm_max >= val)
                    {
                        T tmp1 = val / norm_max;
                        norm_ssq += tmp1 * tmp1;
                    }
                    else
                    {
                        T tmp1 = norm_max / val;
                        T tmp2 = tmp1 * tmp1;
                        norm_ssq = one + norm_ssq*tmp2;
                        norm_max = val;
                    }
                }
            }
            norm_max *= alpha;
            T norm = norm_max * std::sqrt(norm_ssq);
            // Save result
            if(beta == zero)
            {
                norm_dst[dst_offset] = norm;
            }
            else
            {
                norm_dst[dst_offset] = std::hypot(beta*norm_dst[dst_offset],
                        norm);
            }
            ++dst_offset;
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, fp32_t alpha, const fp32_t *src,
        fp32_t beta, fp32_t *norm_dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, fp64_t alpha, const fp64_t *src,
        fp64_t beta, fp64_t *norm_dst)
    noexcept;

} // namespace norm
} // namespace kernel
} // namespace nntile

