/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/scal_inplace.cc
 * Inplace scal of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-02
 * */

#include "nntile/tile/scal_inplace.hh"
#include "nntile/starpu/scal_inplace.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise scal_inplaceing
template<typename T>
void scal_inplace_async(T alpha, const Tile<T> &data)
{
    // Insert task
    starpu::scal_inplace::submit<T>(alpha, data.nelems, data);
}

//! Tile-wise scal_inplaceing
template<typename T>
void scal_inplace(T alpha, const Tile<T> &data)
{
    scal_inplace_async<T>(alpha, data);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void scal_inplace_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &data);

template
void scal_inplace_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &data);

// Explicit instantiation
template
void scal_inplace<fp32_t>(fp32_t alpha, const Tile<fp32_t> &data);

template
void scal_inplace<fp64_t>(fp64_t alpha, const Tile<fp64_t> &data);

} // namespace tile
} // namespace nntile

