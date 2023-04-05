/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/gelutanh_backward.cc
 * Backward approximate GeLU operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-04-05
 * */

#include "nntile/tile/gelutanh_backward.hh"
#include "nntile/starpu/gelutanh_backward.hh"

namespace nntile
{
namespace tile
{

//! Asynchronous tile-wise backward approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise backward approximate GeLU operation
 * */
template<typename T>
void gelutanh_backward_async(const Tile<T> &x, const Tile<T> &dy,
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
    starpu::gelutanh_backward::submit<T>(x.nelems, x, dy, dx);
}

//! Blocking version of tile-wise backward approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise backward approximate GeLU operation
 * */
template<typename T>
void gelutanh_backward(const Tile<T> &x, const Tile<T> &dy, const Tile<T> &dx)
{
    gelutanh_backward_async<T>(x, dy, dx);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void gelutanh_backward_async<fp32_t>(const Tile<fp32_t> &x, const Tile<fp32_t> &dy,
        const Tile<fp32_t> &dx);

template
void gelutanh_backward_async<fp64_t>(const Tile<fp64_t> &x, const Tile<fp64_t> &dy,
        const Tile<fp64_t> &dx);

// Explicit instantiation
template
void gelutanh_backward<fp32_t>(const Tile<fp32_t> &x, const Tile<fp32_t> &dy,
        const Tile<fp32_t> &dx);

template
void gelutanh_backward<fp64_t>(const Tile<fp64_t> &x, const Tile<fp64_t> &dy,
        const Tile<fp64_t> &dx);

} // namespace tile
} // namespace nntile

