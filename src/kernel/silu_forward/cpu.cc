/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/silu_forward/cpu.cc
 * Forward SiLU operation on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/silu_forward/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::silu_forward
{

template<typename T>
void cpu(Index nelems, const T *src, T *dst)
    noexcept
//! Forward SiLU operation on CPU
/*! Does the following per-element operation:
 * dst[i] = src[i] * sigmoid(src[i])
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src: Input array
 * @params[out] dst: Output array
 * */
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < nelems; ++i)
    {
        Y src_val{src[i]};
        dst[i] = static_cast<T>(src_val / (Y{1.} + std::exp(-src_val)));
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp32_fast_tf32_t>(Index nelems, const fp32_fast_tf32_t *src, fp32_fast_tf32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, const bf16_t *src, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::relu_forward
