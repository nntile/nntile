/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/add.cc
 * Add operation for two Tile<T>'s
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-05-08
 * */

#include "nntile/tile/add.hh"
#include "nntile/starpu/add.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise add operation
template<typename T>
void add_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst)
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
    // Do nothing if alpha is zero and beta is one
    if(alpha == 0.0 && beta == 1.0)
    {
        return;
    }
    // Insert corresponding task
    starpu::add::submit<T>(src.nelems, alpha, src, beta, dst);
}

//! Tile-wise add operation
template<typename T>
void add(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst)
{
    add_async<T>(alpha, src, beta, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void add_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &dst);

template
void add_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &dst);

// Explicit instantiation of template
template
void add<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &dst);

template
void add<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &dst);
        
} // namespace tile
} // namespace nntile

