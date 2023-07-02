/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/scal.cc
 * Scal operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-02
 * */

#include "nntile/tile/scal.hh"
#include "nntile/starpu/scal.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise scal operation
template<typename T>
void scal_async(T alpha, const Tile<T> &src, const Tile<T> &dst)
{
    // Check dimensions
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    // Insert corresponding task
    starpu::scal::submit<T>(src.nelems, alpha, src, dst);
}

//! Tile-wise scal operation
template<typename T>
void scal(T alpha, const Tile<T> &src, const Tile<T> &dst)
{
    scal_async<T>(alpha, src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void scal_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void scal_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

// Explicit instantiation of template
template
void scal<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void scal<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);
        
} // namespace tile
} // namespace nntile

