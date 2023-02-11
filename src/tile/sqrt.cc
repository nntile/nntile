/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sqrt.cc
 * Sqrt operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#include "nntile/tile/sqrt.hh"
#include "nntile/starpu/sqrt.hh"

namespace nntile
{
namespace tile
{

//! Asynchronous tile-wise sqrt operation
/*! @param[inout] A: Tile for the element-wise sqrt operation
 * */
template<typename T>
void sqrt_async(const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::sqrt::submit<T>(A.nelems, A);
}

//! Blocking version of tile-wise sqrt operation
/*! @param[inout] A: Tile for the element-wise sqrt operation
 * */
template<typename T>
void sqrt(const Tile<T> &A)
{
    sqrt_async<T>(A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void sqrt_async<fp32_t>(const Tile<fp32_t> &A);

template
void sqrt_async<fp64_t>(const Tile<fp64_t> &A);

// Explicit instantiation
template
void sqrt<fp32_t>(const Tile<fp32_t> &A);

template
void sqrt<fp64_t>(const Tile<fp64_t> &A);

} // namespace tile
} // namespace nntile
