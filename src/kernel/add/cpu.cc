/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add/cpu.cc
 * Add operation on buffers on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::add
{

template<typename T>
void cpu(Index nelems, Scalar alpha, const T* src, Scalar beta, T* dst)
    noexcept
//! Add of two buffers on CPU
/*! Performs the following operation:
 *      dst[i] = alpha*src[i] + beta*dst[i],
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[in] beta: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the add operation
 * */
{
    using Y = typename T::repr_t;
    const Y alpha_{alpha}, beta_{beta};
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = alpha_*Y{src[i]} + beta_*Y{dst[i]};
    }
}

// Explicit instantiation
template
void cpu<fp64_t>(Index nelems, Scalar alpha, const fp64_t* src, Scalar beta,
        fp64_t* dst)
    noexcept;

template
void cpu<fp32_t>(Index nelems, Scalar alpha, const fp32_t* src, Scalar beta,
        fp32_t* dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, Scalar alpha, const bf16_t* src, Scalar beta,
        bf16_t* dst)
    noexcept;

} // namespace nntile::kernel::add
