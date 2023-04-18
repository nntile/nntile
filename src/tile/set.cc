/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/set.cc
 * Set operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#include "nntile/tile/set.hh"
#include "nntile/starpu/set.hh"

namespace nntile
{
namespace tile
{

//! Asynchronous tile-wise set operation
/*! @param[inout] A: Tile for the element-wise set operation
 * */
template<typename T>
void set_async(T val, const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::set::submit<T>(A.nelems, val, A);
}

//! Blocking version of tile-wise set operation
/*! @param[inout] A: Tile for the element-wise set operation
 * */
template<typename T>
void set(T val, const Tile<T> &A)
{
    set_async<T>(val, A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void set_async<fp32_t>(fp32_t val, const Tile<fp32_t> &A);

template
void set_async<fp64_t>(fp64_t val, const Tile<fp64_t> &A);

// Explicit instantiation
template
void set<fp32_t>(fp32_t val, const Tile<fp32_t> &A);

template
void set<fp64_t>(fp64_t val, const Tile<fp64_t> &A);

} // namespace tile
} // namespace nntile

