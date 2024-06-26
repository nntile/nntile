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
 * @version 1.0.0
 * */

#include "nntile/kernel/add/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::add
{

template<typename T>
void cpu(Index nelems, T alpha_, const T* src_, T beta_, T* dst_)
    noexcept
//! Add of two buffers on CPU
/*! Performs the following operation:
 *      dst[i] = alpha*src[i] + beta*dst[i],
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src tensor
 * @param[in] src_: Source tensor
 * @param[in] beta_: Scalar multiplier for the dst tensor
 * @param[inout] dst_: Destination of the add operation
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto *src = reinterpret_cast<const Y *>(src_);
    auto *dst = reinterpret_cast<Y *>(dst_);
    const Y alpha{alpha_}, beta{beta_};
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = alpha*src[i] + beta*dst[i];
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t alpha, const fp32_t* src, fp32_t beta,
        fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t alpha, const fp64_t* src, fp64_t beta,
        fp64_t* dst)
    noexcept;

} // namespace nntile::kernel::add
