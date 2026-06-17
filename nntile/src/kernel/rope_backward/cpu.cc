/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/rope_backward/cpu.cc
 * Backward for Rotary Positional Embedding
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/rope_backward/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::rope_backward
{

template<typename T>
void cpu(Index nrows, Index ncols, const T *sin, const T *cos, const T *dy,
    T *dx) noexcept
{
    using Y = typename T::repr_t;
    const Index m = ncols;
    const Index n = nrows;
    for(Index j = 0; j < n; ++j)
    {
        for(Index i = 0; i < m; ++i)
        {
            const Index l = 2 * (i + j * m);
            Y c{cos[i]}, s{sin[i]};
            Y a{dy[l]}, b{dy[l + 1]};
            dx[l] = static_cast<T>(c * a + s * b);
            dx[l + 1] = static_cast<T>(c * b - s * a);
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nrows, Index ncols, const fp32_t *sin,
    const fp32_t *cos, const fp32_t *dy, fp32_t *dx) noexcept;

template
void cpu<fp64_t>(Index nrows, Index ncols, const fp64_t *sin,
    const fp64_t *cos, const fp64_t *dy, fp64_t *dx) noexcept;

template
void cpu<fp16_t>(Index nrows, Index ncols, const fp16_t *sin,
    const fp16_t *cos, const fp16_t *dy, fp16_t *dx) noexcept;

template
void cpu<bf16_t>(Index nrows, Index ncols, const bf16_t *sin,
    const bf16_t *cos, const bf16_t *dy, bf16_t *dx) noexcept;

} // namespace nntile::kernel::rope_backward
