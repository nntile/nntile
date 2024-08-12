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
void cpu(Index m, Index n, const T *sin, const T *cos, const T *dy, T *dx)
    noexcept
/*! Change provided 2-by-m-by-n src tensor and write result into dst tensor
 *  sin, cos are tensors of shape (m). Each column holds sines and cosines.
 *  TODO: describe math here
 *
 * Although this is backward procedure, which usually implies += update of the
 * output, we specifically use = for the output.
 *
 * @param[in] m: Size of sin and cos tensors
 * @param[in] n: Size of the second mode of src and dst tensors
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] dy: Gradient over output of forward RoPE
 * @param[out] dx: Gradient over input of forward RoPE
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
            Y a{dy[l]}, b{dy[l+1]};
            dx[l] = static_cast<T>(c*a + s*b);
            dx[l+1] = static_cast<T>(c*b - s*a);
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, const fp32_t *sin, const fp32_t *cos,
        const fp32_t *dy, fp32_t *dx)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, const fp64_t *sin, const fp64_t *cos,
        const fp64_t *dy, fp64_t *dx)
    noexcept;

template
void cpu<fp32_fast_tf32_t>(Index m, Index n, const fp32_fast_tf32_t *sin,
        const fp32_fast_tf32_t *cos, const fp32_fast_tf32_t *dy,
        fp32_fast_tf32_t *dx)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, const bf16_t *sin, const bf16_t *cos,
        const bf16_t *dy, bf16_t *dx)
    noexcept;

} // namespace rope_backward
