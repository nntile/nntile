/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm/cpu.cc
 * Euclidean norm of all elements in a buffer on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/norm/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::norm
{

template<typename T>
void cpu(Index nelems, Scalar alpha_, const T *src, Scalar beta_, T *dst)
    noexcept
//! Euclidean norm of all elements in a buffer (out-of-place version)
/*! For a provided array src of nelems elements compute the Euclidean norm
 * and combine it with the existing dst[0] value.
 * Mnemonically, the following operations are performed:
 *      dst[0] = hypot(alpha * norm(src[...]), beta * dst[0])
 *
 * @param[in] nelems: Number of elements in src array
 * @param[in] alpha_: Scaling factor for the norm
 * @param[in] src: Input contiguous array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Output scalar (single element array)
 * */
{
    using Y = typename T::repr_t;
    Y alpha = static_cast<Y>(alpha_), beta = static_cast<Y>(beta_);
    constexpr Y zero = 0.0, one = 1.0;
    alpha = std::fabs(alpha);
    // Init norm computation
    Y norm_max = zero, norm_ssq = zero, c = zero, y, t;
    // Cycle over all elements to compute the norm
    for(Index i = 0; i < nelems; ++i)
    {
        // Read value from source
        Y src_val = static_cast<Y>(src[i]);
        Y val = std::fabs(src_val);
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
    // Compute the norm value
    Y norm_val = alpha * norm_max * std::sqrt(norm_ssq);
    // Apply beta scaling to destination and add
    Y dst_val = static_cast<Y>(dst[0]);
    if(beta == zero)
    {
        dst[0] = norm_val;
    }
    else if(norm_val > 0)
    {
        dst[0] = std::hypot(beta * dst_val, norm_val);
    }
    else
    {
        dst[0] = std::fabs(beta * dst_val);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, Scalar alpha, const fp32_t *src, Scalar beta,
        fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, Scalar alpha, const fp64_t *src, Scalar beta,
        fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, Scalar alpha, const bf16_t *src, Scalar beta,
        bf16_t *dst)
    noexcept;

template
void cpu<fp16_t>(Index nelems, Scalar alpha, const fp16_t *src, Scalar beta,
        fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::norm
