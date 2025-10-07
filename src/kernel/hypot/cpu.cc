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
void cpu(Index nelems, Scalar alpha_, const T* src1, Scalar beta_, const T* src2, T* dst)
    noexcept
//! hypot of two buffers on CPU
/*! Performs the following operation:
 *      dst[i] = hypot(alpha*src1[i], beta*src2[i]),
 * where alpha and beta are scalars.
 *
 * @param[in] nelems: Size of the src1, src2 and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src1 tensor
 * @param[in] src1: First source tensor
 * @param[in] beta_: Scalar multiplier for the src2 tensor
 * @param[in] src2: Second source tensor
 * @param[out] dst: Destination of the hypot operation
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
                dst[i] = zero;
            }
        }
        else
        {
            for(Index i = 0; i < nelems; ++i)
            {
                Y src2_val = static_cast<Y>(src2[i]);
                dst[i] = std::fabs(beta * src2_val);
            }
        }
    }
    else
    {
        if(beta == zero)
        {
            for(Index i = 0; i < nelems; ++i)
            {
                Y src1_val = static_cast<Y>(src1[i]);
                dst[i] = std::fabs(alpha * src1_val);
            }
        }
        else
        {
            for(Index i = 0; i < nelems; ++i)
            {
                Y src1_val = static_cast<Y>(src1[i]);
                Y src2_val = static_cast<Y>(src2[i]);
                dst[i] = std::hypot(alpha*src1_val, beta*src2_val);
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, Scalar alpha, const fp32_t* src1, Scalar beta, const fp32_t* src2, fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, Scalar alpha, const fp64_t* src1, Scalar beta, const fp64_t* src2, fp64_t* dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, Scalar alpha, const bf16_t* src1, Scalar beta, const bf16_t* src2, bf16_t* dst)
    noexcept;

template
void cpu<fp16_t>(Index nelems, Scalar alpha, const fp16_t* src1, Scalar beta, const fp16_t* src2, fp16_t* dst)
    noexcept;

} // namespace nntile::kernel::hypot
