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
void cpu(Index ncols, Index nrows, const T *sin, const T *cos, const T *src,
    T *dst) noexcept
/*! Apply RoPE on the Fortran-order (2, m, n) view of flat tile data.
 *
 * @param[in] ncols: sin/cos tile extent (historical m)
 * @param[in] nrows: spatial tile extent (historical n)
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
void cpu<fp32_t>(Index ncols, Index nrows, const fp32_t *sin,
    const fp32_t *cos, const fp32_t *src, fp32_t *dst) noexcept;

template
void cpu<fp64_t>(Index ncols, Index nrows, const fp64_t *sin,
    const fp64_t *cos, const fp64_t *src, fp64_t *dst) noexcept;

template
void cpu<fp16_t>(Index ncols, Index nrows, const fp16_t *sin,
    const fp16_t *cos, const fp16_t *src, fp16_t *dst) noexcept;

template
void cpu<bf16_t>(Index ncols, Index nrows, const bf16_t *sin,
    const bf16_t *cos, const bf16_t *src, bf16_t *dst) noexcept;

} // namespace nntile::kernel::rope
