/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/silu_backward.cc
 * Backward SiLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/silu_backward.hh"
#include "nntile/starpu/silu_backward.hh"

namespace nntile::tile
{

//! Asynchronous tile-wise backward SiLU operation
/*! @param[inout] A: Tile for the element-wise backward SiLU operation
 * */
template<typename T>
void silu_backward_async(const Tile<T> &x, const Tile<T> &dy,
        const Tile<T> &dx)
{
    // Check shapes
    if(x.shape != dy.shape)
    {
        throw std::runtime_error("x.shape != dy.shape");
    }
    if(x.shape != dx.shape)
    {
        throw std::runtime_error("x.shape != dx.shape");
    }
    // Submit task without any arguments checked
    starpu::silu_backward.submit<std::tuple<T>>(x.nelems, x, dy, dx);
}

//! Blocking version of tile-wise backward SiLU operation
/*! @param[inout] A: Tile for the element-wise backward SiLU operation
 * */
template<typename T>
void silu_backward(const Tile<T> &x, const Tile<T> &dy, const Tile<T> &dx)
{
    silu_backward_async<T>(x, dy, dx);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void silu_backward_async<fp32_t>(const Tile<fp32_t> &x, const Tile<fp32_t> &dy,
        const Tile<fp32_t> &dx);

template
void silu_backward_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &x,
        const Tile<fp32_fast_tf32_t> &dy, const Tile<fp32_fast_tf32_t> &dx);

template
void silu_backward_async<fp64_t>(const Tile<fp64_t> &x, const Tile<fp64_t> &dy,
        const Tile<fp64_t> &dx);

template
void silu_backward_async<bf16_t>(const Tile<bf16_t> &x, const Tile<bf16_t> &dy,
        const Tile<bf16_t> &dx);

// Explicit instantiation
template
void silu_backward<fp32_t>(const Tile<fp32_t> &x, const Tile<fp32_t> &dy,
        const Tile<fp32_t> &dx);

template
void silu_backward<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &x,
        const Tile<fp32_fast_tf32_t> &dy, const Tile<fp32_fast_tf32_t> &dx);

template
void silu_backward<fp64_t>(const Tile<fp64_t> &x, const Tile<fp64_t> &dy,
        const Tile<fp64_t> &dx);

template
void silu_backward<bf16_t>(const Tile<bf16_t> &x, const Tile<bf16_t> &dy,
        const Tile<bf16_t> &dx);

} // namespace nntile::tile
