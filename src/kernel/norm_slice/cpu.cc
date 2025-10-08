/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm_slice/cpu.cc
 * Euclidean norms of fibers into a slice of a buffer on CPU (out-of-place version)
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/norm_slice/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::norm_slice
{

template<typename T>
void cpu(Index m, Index n, Index k, Scalar alpha_, const T *src, Scalar beta_,
        const T *dst, T *result)
    noexcept
//! Euclidean norms over fibers along middle axis into a slice of a tensor (out-of-place version)
/*! For a provided m-by-k-by-n input array src compute norms of fibers
 * along second axis with k elements, resulting in m-by-n output array-slice
 * result.
 * Mnemonically, the following operations are performed:
 *      result[i,j] = hypot(beta*dst[i,j], alpha*norm(src[i,:,j]))
 *
 * @param[in] m: Size of the first mode of src, dst and result arrays
 * @param[in] n: Size of the last mode of src, dst and result arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha_: Scaling factor for src
 * @param[in] src_: Input contiguous m-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[in] dst_: Input contiguous m-by-n array
 * @param[out] result_: Output contiguous m-by-n array, that contains norms
 *      along middle axis combined with dst values.
 * */
{
    using Y = typename T::repr_t;
    Y alpha{alpha_}, beta{beta_};
    const Index mk = m * k;
    constexpr Y zero{0.0}, one{1.0};
    alpha = std::fabs(alpha);
    // Cycle over column of the output buffer result
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over row of the output buffer result
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Pointer to a corresponding fiber of the source array src
            const T *src_fiber = src + i2*mk + i1;
            // Init norm of the fiber
            Y norm_max{zero}, norm_ssq{zero}, c{zero}, y, t;
            // Cycle over fiber elements and accumulate the norm
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                Y val = std::fabs(Y{src_fiber[i0*m]});
                // Update norm only if new value is non-zero
                if(val > 0)
                {
                    if(norm_max >= val)
                    {
                        Y tmp1 = val / norm_max;
                        //norm_ssq += tmp1 * tmp1;
                        y = tmp1*tmp1 - c;
                        t = norm_ssq + y;
                        c = (t-norm_ssq) - y;
                        norm_ssq = t;
                    }
                    else
                    {
                        Y tmp1 = norm_max / val;
                        Y tmp2 = tmp1 * tmp1;
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
            // Get the scaled dst value
            Y dst_val = beta * Y{dst[i2*m+i1]};
            // Compute the result using hypot function
            if(beta == zero)
            {
                // result = norm_max * sqrt(norm_ssq)
                result[i2*m+i1] = static_cast<T>(norm_max * std::sqrt(norm_ssq));
            }
            else if(norm_max > 0)
            {
                // result = hypot(dst_val, norm)
                Y tmp_dst = std::fabs(dst_val);
                if(norm_max >= tmp_dst)
                {
                    Y tmp1 = tmp_dst / norm_max;
                    result[i2*m+i1] = static_cast<T>(norm_max * std::sqrt((tmp1*tmp1-c)+norm_ssq));
                }
                else
                {
                    Y tmp1 = norm_max / tmp_dst;
                    Y tmp2 = tmp1 * tmp1;
                    c *= tmp2;
                    norm_ssq *= tmp2;
                    result[i2*m+i1] = static_cast<T>(tmp_dst * std::sqrt((one-c)+norm_ssq));
                }
            }
            // norm_max==0
            else
            {
                result[i2*m+i1] = static_cast<T>(std::fabs(dst_val));
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Scalar alpha, const fp32_t *src,
        Scalar beta, const fp32_t *dst, fp32_t *result)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Scalar alpha, const fp64_t *src,
        Scalar beta, const fp64_t *dst, fp64_t *result)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, Scalar alpha, const bf16_t *src,
        Scalar beta, const bf16_t *dst, bf16_t *result)
    noexcept;

template
void cpu<fp16_t>(Index m, Index n, Index k, Scalar alpha, const fp16_t *src,
        Scalar beta, const fp16_t *dst, fp16_t *result)
    noexcept;

} // namespace nntile::kernel::norm_slice