/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm_slice/cpu.cc
 * Euclidean norms of fibers into a slice of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-07
 * */

#include "nntile/kernel/norm_slice/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace norm_slice
{

template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src, T beta, T *dst)
    noexcept
//! Euclidean norms over fibers along middle axis into a slice of a tensor
/*! For a provided m-by-k-by-n input array src compute norms of fibers
 * along second axis with k elements, resulting in m-by-n output array-slice
 * dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = hypot(beta*dst[i,j], alpha*norm(src[i,:,j]))
 *
 * @param[in] m: Size of the first mode of src and dst arrays
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-n array, that
 *      accumulates norms along middle axis.
 * */
{
    const Index mk = m * k;
    constexpr T zero = 0.0, one = 1.0;
    alpha = std::abs(alpha);
    // Cycle over column of the output buffer dst
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over row of the output buffer dst
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Pointer to a corresponding fiber of the source array src
            const T *src_fiber = src + i2*mk + i1;
            // Init norm of the fiber
            T norm_max = zero, norm_ssq = zero, c = zero, y, t;
            // Output value
            T &result = dst[i2*m+i1];
            // Cycle over fiber elements and accumulate the norm
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = std::abs(src_fiber[i0*m]);
                // Update norm only if new value is non-zero
                if(val > 0)
                {
                    if(norm_max >= val)
                    {
                        T tmp1 = val / norm_max;
                        //norm_ssq += tmp1 * tmp1;
                        y = tmp1*tmp1 - c;
                        t = norm_ssq + y;
                        c = (t-norm_ssq) - y;
                        norm_ssq = t;
                    }
                    else
                    {
                        T tmp1 = norm_max / val;
                        T tmp2 = tmp1 * tmp1;
                        y = one - c*tmp2;
                        norm_ssq *= tmp2;
                        t = norm_ssq + y;
                        c = (t-norm_ssq) - y;
                        norm_ssq = t;
                        norm_max = val;
                    }
                }
            }
            // Get the scaled norm
            norm_max *= alpha;
            //T norm = norm_max * std::sqrt(norm_ssq);
            // Update output value
            if(beta == zero)
            {
                //result = norm;
                result = norm_max * std::sqrt(norm_ssq);
            }
            else if(norm_max > 0)
            {
                //result = std::hypot(beta*result, norm);
                T tmp_res = std::abs(beta * result);
                if(norm_max >= tmp_res)
                {
                    T tmp1 = tmp_res / norm_max;
                    result = norm_max * std::sqrt((tmp1*tmp1-c)+norm_ssq);
                }
                else
                {
                    T tmp1 = norm_max / tmp_res;
                    T tmp2 = tmp1 * tmp1;
                    c *= tmp2;
                    norm_ssq *= tmp2;
                    result = tmp_res * std::sqrt((one-c)+norm_ssq);
                }
            }
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

} // namespace norm_slice
} // namespace kernel
} // namespace nntile

