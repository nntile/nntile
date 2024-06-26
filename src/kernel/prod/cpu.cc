/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/prod/cpu.cc
 * Per-element product of two buffers on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/prod/cpu.hh"

namespace nntile::kernel::prod
{

template<typename T>
void cpu(Index nelems, const T *src_T, T *dst_T)
    noexcept
//! Per-element product of two buffers
/*! One of the buffers serves as output
 *
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] src: Input buffer
 * @param[inout] dst: Input buffers that contains output in the end
 * */
{
    using Y = typename T::internal_t;
    auto src = reinterpret_cast<const Y *>(src_T);
    auto dst = reinterpret_cast<Y *>(dst_T);
    // Cycle over buffers
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] *= src[i];
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *src, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::prod
