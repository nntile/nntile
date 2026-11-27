/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/kernel/swap_two_axes/cpu.cc
 * Swap axes 1 and 3 in a 5D buffer on CPU.
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/swap_two_axes/cpu.hh"

namespace nntile::kernel::swap_two_axes
{

template<typename T>
void cpu(
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const T *src,
    T *dst) noexcept
{
    for (Index i0 = 0; i0 < d0; ++i0)
    {
        for (Index i1 = 0; i1 < d1; ++i1)
        {
            for (Index i2 = 0; i2 < d2; ++i2)
            {
                for (Index i3 = 0; i3 < d3; ++i3)
                {
                    for (Index i4 = 0; i4 < d4; ++i4)
                    {
                        const Index src_idx =
                            ((((i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4 +
                                i4);
                        const Index dst_idx =
                            ((((i0 * d3 + i3) * d2 + i2) * d1 + i1) * d4 +
                                i4);
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

template
void cpu<fp32_t>(
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const fp32_t *src,
    fp32_t *dst) noexcept;

template
void cpu<fp64_t>(
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const fp64_t *src,
    fp64_t *dst) noexcept;

template
void cpu<bf16_t>(
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const bf16_t *src,
    bf16_t *dst) noexcept;

template
void cpu<fp16_t>(
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const fp16_t *src,
    fp16_t *dst) noexcept;

} // namespace nntile::kernel::swap_two_axes
