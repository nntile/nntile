/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/maximum/cpu.cc
 * Per-element maximum of two buffers on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/maximum/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::maximum
{

template<typename T>
void cpu(Index nelems, const T *src, T *dst)
    noexcept
//! Per-element maximum of two buffers
/*! One of the buffers serves as output
 *
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] src_: Input buffer
 * @param[inout] dst_: Input buffers that contains output in the end
 * */
{
    using Y = typename T::repr_t;
    // Cycle over buffers
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = T{std::fmax(static_cast<Y>(src[i]), static_cast<Y>(dst[i]))};
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, const bf16_t *src, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::maximum
