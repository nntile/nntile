/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/hypot/cpu.cc
 * hypot operation on buffers on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/hypot/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::hypot
{

template<typename T>
void cpu(Index nelems, Scalar alpha_, const T* src, Scalar beta_, T* dst)
    noexcept
//! hypot of two buffers on CPU
/*! Performs the following operation:
 *      dst[i] = hypot(alpha*src[i], beta*dst[i]),
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[in] beta_: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the hypot operation
 * */
{
    using Y = typename T::repr_t;
    const Y zero{0.0}, alpha{alpha_}, beta{beta_};
    if(alpha == zero)
    {
        if(beta == zero)
        {
            for(Index i = 0; i < nelems; ++i)
            {
                dst[i] = static_cast<T>(zero);
            }
        }
        else
        {
            for(Index i = 0; i < nelems; ++i)
            {
                dst[i] = static_cast<T>(std::fabs(beta * static_cast<Y>(dst[i])));
            }
        }
    }
    else
    {
        if(beta == zero)
        {
            for(Index i = 0; i < nelems; ++i)
            {
                dst[i] = static_cast<T>(std::fabs(alpha * static_cast<Y>(src[i])));
            }
        }
        else
        {
            for(Index i = 0; i < nelems; ++i)
            {
                dst[i] = static_cast<T>(std::hypot(alpha*static_cast<Y>(src[i]),
                                                   beta*static_cast<Y>(dst[i])));
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, Scalar alpha, const fp32_t* src, Scalar beta,
        fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, Scalar alpha, const fp64_t* src, Scalar beta,
        fp64_t* dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, Scalar alpha, const bf16_t* src, Scalar beta,
        bf16_t* dst)
    noexcept;

} // namespace nntile::kernel::hypot
