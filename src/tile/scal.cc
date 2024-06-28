/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/scal.cc
 * Scal operation for Tile<T>
 *
 * @version 1.0.0
 * */

#include "nntile/tile/scal.hh"
#include "nntile/starpu/scal.hh"

namespace nntile::tile
{

//! Tile-wise scal operation
template<typename T>
void scal_async(scal_t alpha, const Tile<T> &src, const Tile<T> &dst)
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
void scal(scal_t alpha, const Tile<T> &src, const Tile<T> &dst)
{
    scal_async<T>(alpha, src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void scal_async<fp32_t>(scal_t alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void scal_async<fp32_fast_tf32_t>(scal_t alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void scal_async<fp64_t>(scal_t alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

// Explicit instantiation of template
template
void scal<fp32_t>(scal_t alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void scal<fp32_fast_tf32_t>(scal_t alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void scal<fp64_t>(scal_t alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

} // namespace nntile::tile
