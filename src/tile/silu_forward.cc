/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/silu_forward.cc
 * Forward SiLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/silu_forward.hh"
#include "nntile/starpu/silu_forward.hh"

namespace nntile::tile
{

template<typename T>
void silu_forward_async(const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Submit forward relu
    starpu::silu_forward::submit<T>(src.nelems, src, dst);
}

template<typename T>
void silu_forward(const Tile<T> &src, const Tile<T> &dst)
{
    silu_forward_async<T>(src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void silu_forward_async<fp32_t>(const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void silu_forward_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void silu_forward_async<fp64_t>(const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void silu_forward_async<bf16_t>(const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

// Explicit instantiation
template
void silu_forward<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void silu_forward<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void silu_forward<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void silu_forward<bf16_t>(const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

} // namespace nntile::tile
