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
void cpu(Index m, Index n, Index k, Scalar alpha_, const T *src1, Scalar beta_,
        const T *src2, T *dst)
    noexcept
//! Euclidean norms over fibers along middle axis into a slice of a tensor (out-of-place version)
/*! For a provided m-by-k-by-n input array src1 compute norms of fibers
 * along second axis with k elements, resulting in m-by-n output array-slice
 * dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = hypot(beta*src2[i,j], alpha*norm(src1[i,:,j]))
 *
 * @param[in] m: Size of the first mode of src1, src2 and dst arrays
 * @param[in] n: Size of the last mode of src1, src2 and dst arrays
 * @param[in] k: Size of the middle mode of src1 array
 * @param[in] alpha_: Scaling factor for src1
 * @param[in] src1_: Input contiguous m-by-k-by-n array
 * @param[in] beta_: Scaling factor for src2
 * @param[in] src2_: Input contiguous m-by-n array
 * @param[out] dst_: Output contiguous m-by-n array, that contains norms
 *      along middle axis combined with src2 values.
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
            // Pointer to a corresponding fiber of the source array src1
            const T *src1_fiber = src1 + i2*mk + i1;
            // Init norm of the fiber
            Y norm_max{zero}, norm_ssq{zero}, c{zero}, y, t;
            // Cycle over fiber elements and accumulate the norm
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                Y val = std::fabs(Y{src1_fiber[i0*m]});
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
            // Get the scaled src2 value
            Y src2_val = beta * Y{src2[i2*m+i1]};
            // Compute the result using hypot function
            if(beta == zero)
            {
                // dst = norm_max * sqrt(norm_ssq)
                dst[i2*m+i1] = static_cast<T>(norm_max * std::sqrt(norm_ssq));
            }
            else if(norm_max > 0)
            {
                // result = hypot(src2_val, norm)
                Y tmp_src2 = std::fabs(src2_val);
                if(norm_max >= tmp_src2)
                {
                    Y tmp1 = tmp_src2 / norm_max;
                    dst[i2*m+i1] = static_cast<T>(norm_max * std::sqrt((tmp1*tmp1-c)+norm_ssq));
                }
                else
                {
                    Y tmp1 = norm_max / tmp_src2;
                    Y tmp2 = tmp1 * tmp1;
                    c *= tmp2;
                    norm_ssq *= tmp2;
                    dst[i2*m+i1] = static_cast<T>(tmp_src2 * std::sqrt((one-c)+norm_ssq));
                }
            }
            // norm_max==0
            else
            {
                dst[i2*m+i1] = static_cast<T>(std::fabs(src2_val));
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Scalar alpha, const fp32_t *src1,
        Scalar beta, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Scalar alpha, const fp64_t *src1,
        Scalar beta, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, Scalar alpha, const bf16_t *src1,
        Scalar beta, const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cpu<fp16_t>(Index m, Index n, Index k, Scalar alpha, const fp16_t *src1,
        Scalar beta, const fp16_t *src2, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::norm_slice
