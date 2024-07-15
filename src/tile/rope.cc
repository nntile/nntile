/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/rope.cc
 * Tile wrappers for the Rotary Positional Embedding
 *
 * @version 1.0.0
 * @author Gleb Karpov
 * @date 2024-06-29
 * */

#include "nntile/tile/rope.hh"
#include "nntile/starpu/rope.hh"

namespace nntile
{
namespace tile
{

template<typename T>
void rope_async(const Tile<T> &sin, const Tile<T> &cos, 
        const Tile<T> &src, const Tile<T> &dst, Index axis)
//! Tile<T> Rotary Positional Embedding
/*! Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{   Index two = 2;
    // Check dimensions
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }

    if(sin.ndim != cos.ndim)
    {
        throw std::runtime_error("sin.ndim != cos.ndim");
    }

    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes of tiles
    
    if(src.shape[0] != two * sin.shape[0])
    {
        throw std::runtime_error("dst.shape[0] != 2 * sin.shape[0]");
    }

    if(src.shape[0] != two * cos.shape[0])
    {
        throw std::runtime_error("dst.shape[0] != 2 * cos.shape[0]");
    }

    for(Index i = 0; i < sin.ndim; ++i)
    {
        if(sin.shape[i] != cos.shape[i])
        {
            throw std::runtime_error("sin.shape[i] != cos.shape[i]");
        }
    }
    if(dst.shape != src.shape)
    {
        throw std::runtime_error("dst.shape != src2.shape");
    }
    // Reshape inputs for simplicity: src -> (m,n), dst -> (m,k,n)
    Index m, k, l;
    m = sin.matrix_shape[1][0];
    k = sin.matrix_shape[1][1];
    l = dst.matrix_shape[1][1];
    // Insert corresponding task
    starpu::rope::submit<T>(m, k, l, sin, cos, src, dst);
}

template<typename T>
void rope(const Tile<T> &sin, const Tile<T> &cos, 
        const Tile<T> &src, const Tile<T> &dst, Index axis)
//! Tile<T> addition of a tensor and a broadcasted slice
/*! Blocking version of rope_async<T>.
 *
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    rope_async<T>(sin, cos, src, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void rope_async<fp32_t>(const Tile<fp32_t> &sin, const Tile<fp32_t> &cos, 
        const Tile<fp32_t> &src, const Tile<fp32_t> &dst, Index axis);

template
void rope_async<fp64_t>(const Tile<fp64_t> &sin, const Tile<fp64_t> &cos, 
        const Tile<fp64_t> &src, const Tile<fp64_t> &dst, Index axis);

// Explicit instantiation of template
template
void rope<fp32_t>(const Tile<fp32_t> &sin, const Tile<fp32_t> &cos, 
        const Tile<fp32_t> &src, const Tile<fp32_t> &dst, Index axis);

template
void rope<fp64_t>(const Tile<fp64_t> &sin, const Tile<fp64_t> &cos, 
        const Tile<fp64_t> &src, const Tile<fp64_t> &dst, Index axis);

} // namespace tile
} // namespace nntile