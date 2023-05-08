/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/scal.cc
 * Scaling of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-03-29
 * */

#include "nntile/tile/scal.hh"
#include "nntile/starpu/scal.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise scaling
template<typename T>
void scal_async(T alpha, const Tile<T> &data)
{
    // Insert task
    starpu::scal::submit<T>(alpha, data.nelems, data);
}

//! Tile-wise scaling
template<typename T>
void scal(T alpha, const Tile<T> &data)
{
    scal_async<T>(alpha, data);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void scal_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &data);

template
void scal_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &data);

// Explicit instantiation
template
void scal<fp32_t>(fp32_t alpha, const Tile<fp32_t> &data);

template
void scal<fp64_t>(fp64_t alpha, const Tile<fp64_t> &data);

} // namespace tile
} // namespace nntile

