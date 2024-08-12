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
void cpu(Index m, Index n, const T *sin, const T *cos, const T *src, T *dst)
    noexcept
/*! Change provided 2-by-m-by-n src tensor and write result into dst tensor
 *  sin, cos are tensors of shape (m). Each column holds sines and cosines.
 *  dst[2i,j] = cos[i] * src[2i,j] - sin[i] * src[2i+1,j]
 *  dst[2i+1,j] = sin[i] * src[2i,j] + cos[i] * src[2i+1,j]
 *
 * @param[in] m: Size of sin and cos tensors
 * @param[in] n: Size of the second mode of src and dst tensors
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    using Y = typename T::repr_t;
    // Use these angles for pairwise rotation of the same elements across all
    // batches
    for (Index j = 0; j < n; ++j)
    {
        // Cycle over all elements of sin and cos buffers.
        for(Index i = 0; i < m; ++i)
        {
            Index l = 2 * (i+j*m);
            Y c{cos[i]}, s{sin[i]};
            Y a{src[l]}, b{src[l+1]};
            dst[l] = static_cast<T>(c*a - s*b);
            dst[l+1] = static_cast<T>(s*a + c*b);
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, const fp32_t *sin, const fp32_t *cos,
        const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, const fp64_t *sin, const fp64_t *cos,
        const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cpu<fp32_fast_tf32_t>(Index m, Index n, const fp32_fast_tf32_t *sin,
        const fp32_fast_tf32_t *cos, const fp32_fast_tf32_t *src,
        fp32_fast_tf32_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, const bf16_t *sin, const bf16_t *cos,
        const bf16_t *src, bf16_t *dst)
    noexcept;

} // namespace rope
