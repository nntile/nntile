/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sum/cpu.cc
 * Sum all elements of a buffer on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sum/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::sum
{

template<typename T>
void cpu(Index nelems, Scalar alpha_, const T *src, Scalar beta_, T *dst)
    noexcept
//! Sum all elements of a tensor into a scalar
/*! For a provided input array of nelems elements computes the sum of all
 * elements, resulting in a single scalar output.
 * Mnemonically, the following operations are performed:
 *      dst[0] = beta*dst[0] + alpha*sum(src[:])
 *
 * @param[in] nelems: Number of elements in the input array
 * @param[in] alpha_: Scaling factor for src
 * @param[in] src_: Input contiguous array of nelems elements
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst_: Output scalar value
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_}, beta{beta_};
    constexpr Y zero{0.0};
    // Init sum with Kahan summation
    Y sum = zero, c = zero;
    // Cycle over all elements and accumulate the sum using Kahan summation
    for(Index i = 0; i < nelems; ++i)
    {
        Y y = Y{src[i]} - c;
        Y t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    // Update output value
    if(beta == zero)
    {
        dst[0] = static_cast<T>(alpha * sum);
    }
    else
    {
        dst[0] = static_cast<T>((beta * Y{dst[0]} - alpha*c) + alpha*sum);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, Scalar alpha, const fp32_t *src,
        Scalar beta, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, Scalar alpha, const fp64_t *src,
        Scalar beta, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, Scalar alpha, const bf16_t *src,
        Scalar beta, bf16_t *dst)
    noexcept;

template
void cpu<fp16_t>(Index nelems, Scalar alpha, const fp16_t *src,
        Scalar beta, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::sum
