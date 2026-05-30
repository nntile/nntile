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
void cpu(
    Index nelems,
    Scalar alpha,
    const T *src1,
    Scalar beta,
    const T *src2,
    T *dst
) noexcept
//! Add two buffers with optional scaling on CPU
/*! Performs the following operation:
 * dst[i] = alpha*src1[i] + beta*src2[i]
 *
 * This function reads both src1 and src2 even if alpha or beta is zero.
 * If alpha is zero and src1[i] is NaN, then dst[i] will be NaN.
 * If beta is zero and src2[i] is NaN, then dst[i] will be NaN.
 * If such behaviour is not desired, then in a case of alpha or beta being
 * zero, use nntile::kernel::scale instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale
 * @see nntile::kernel::clear
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src1 tensor
 * @param[in] src1: First source tensor
 * @param[in] beta: Scalar multiplier for the src2 tensor
 * @param[in] src2: Second source tensor
 * @param[out] dst: Destination tensor
 * */
{
    using Y = typename T::repr_t;
    const Y alpha_ = alpha, beta_ = beta;
    for(Index i = 0; i < nelems; ++i)
    {
        const Y src1_val = static_cast<Y>(src1[i]);
        const Y src2_val = static_cast<Y>(src2[i]);
        dst[i] = static_cast<T>(alpha_ * src1_val + beta_ * src2_val);
    }
}

// Explicit instantiation
template
void cpu<fp64_t>(Index nelems, Scalar alpha, const fp64_t *src1,
        Scalar beta, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cpu<fp32_t>(Index nelems, Scalar alpha, const fp32_t *src1,
        Scalar beta, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, Scalar alpha, const bf16_t* src1,
        Scalar beta, const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cpu<fp16_t>(Index nelems, Scalar alpha, const fp16_t* src1,
        Scalar beta, const fp16_t *src2, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::add
