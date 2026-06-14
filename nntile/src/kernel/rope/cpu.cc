/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/rope/cpu.cc
 * Rotary Positional Embedding
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/rope/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::rope
{

template<typename T>
void cpu(Index nrows, Index ncols, const T *sin, const T *cos, const T *src,
    T *dst) noexcept
/*! Apply RoPE on a Fortran-order (2, m, n) view of flat C-order tile data.
 *
 * nrows/ncols are swapped vs the logical slow/pair axes: the loop uses
 * m = ncols (pair count) and n = nrows (slow extent).
 *
 * @param[in] nrows: Slow-axis tile extent (Fortran matrix rows)
 * @param[in] ncols: Pair count along the RoPE axis (Fortran matrix cols)
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
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
            Y a{src[l]}, b{src[l + 1]};
            dst[l] = static_cast<T>(c * a - s * b);
            dst[l + 1] = static_cast<T>(s * a + c * b);
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nrows, Index ncols, const fp32_t *sin,
    const fp32_t *cos, const fp32_t *src, fp32_t *dst) noexcept;

template
void cpu<fp64_t>(Index nrows, Index ncols, const fp64_t *sin,
    const fp64_t *cos, const fp64_t *src, fp64_t *dst) noexcept;

template
void cpu<fp16_t>(Index nrows, Index ncols, const fp16_t *sin,
    const fp16_t *cos, const fp16_t *src, fp16_t *dst) noexcept;

template
void cpu<bf16_t>(Index nrows, Index ncols, const bf16_t *sin,
    const bf16_t *cos, const bf16_t *src, bf16_t *dst) noexcept;

} // namespace nntile::kernel::rope
