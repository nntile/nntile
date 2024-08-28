/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/rope.cc
 * Tile wrappers for the Rotary Positional Embedding
 *
 * @version 1.1.0
 * */

#include "nntile/tile/rope.hh"
#include "nntile/starpu/rope.hh"

namespace nntile::tile
{

template<typename T>
void rope_async(const Tile<T> &sin, const Tile<T> &cos, const Tile<T> &src,
        const Tile<T> &dst)
//! Tile<T> Rotary Positional Embedding
/*! Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    // Check dimensions
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }

    if(sin.ndim != cos.ndim)
    {
        throw std::runtime_error("sin.ndim != cos.ndim");
    }

    if(src.ndim < sin.ndim)
    {
        throw std::runtime_error("src.ndim < sin.ndim");
    }

    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }

    if(sin.shape != cos.shape)
    {
        throw std::runtime_error("sin.shape != cos.shape");
    }

    if(sin.ndim == 0)
    {
        throw std::runtime_error("sin.ndim == 0");
    }

    // 0-th dimension is the head_size, which is halved for sin and cos
    if(src.shape[0] != 2*sin.shape[0])
    {
        throw std::runtime_error("src.shape[0] != 2*sin.shape[0]");
    }

    for(Index i = 1; i < sin.ndim; ++i)
    {
        if(src.shape[i] != sin.shape[i])
        {
            throw std::runtime_error("src.shape[i] != sin.shape[i]");
        }
    }

    // Reshape inputs for simplicity: sin,cos -> (m), src,dst -> (2,m,n)
    Index m{sin.nelems}, n={src.matrix_shape[sin.ndim-1][1]};
    // Insert corresponding task
    starpu::rope::submit<T>(m, n, sin, cos, src, dst);
}

template<typename T>
void rope(const Tile<T> &sin, const Tile<T> &cos, const Tile<T> &src,
        const Tile<T> &dst)
//! Tile<T> addition of a tensor and a broadcasted slice
/*! Blocking version of rope_async<T>.
 *
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    rope_async<T>(sin, cos, src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void rope_async<fp32_t>(const Tile<fp32_t> &sin, const Tile<fp32_t> &cos,
        const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void rope_async<fp64_t>(const Tile<fp64_t> &sin, const Tile<fp64_t> &cos,
        const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void rope_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &sin,
        const Tile<fp32_fast_tf32_t> &cos,
        const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void rope_async<bf16_t>(const Tile<bf16_t> &sin, const Tile<bf16_t> &cos,
        const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

// Explicit instantiation of template
template
void rope<fp32_t>(const Tile<fp32_t> &sin, const Tile<fp32_t> &cos,
        const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void rope<fp64_t>(const Tile<fp64_t> &sin, const Tile<fp64_t> &cos,
        const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void rope<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &sin,
        const Tile<fp32_fast_tf32_t> &cos,
        const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void rope<bf16_t>(const Tile<bf16_t> &sin, const Tile<bf16_t> &cos,
        const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

} // namespace nntile::tile
