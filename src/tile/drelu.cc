/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/drelu.cc
 * Derivative of ReLU operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-31
 * */

#include "nntile/tile/drelu.hh"
#include "nntile/starpu/drelu.hh"

namespace nntile
{
namespace tile
{

//! Blocking version of tile-wise derivative of ReLU operation
/*! @param[inout] A: Tile for the element-wise derivative of ReLU operation
 * */
template<typename T>
void drelu_async(const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::drelu::submit<T>(A.nelems, A);
}

//! Blocking version of tile-wise derivative of ReLU operation
/*! @param[inout] A: Tile for the element-wise derivative of ReLU operation
 * */
template<typename T>
void drelu(const Tile<T> &A)
{
    drelu_async<T>(A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void drelu_async<fp32_t>(const Tile<fp32_t> &A);

template
void drelu_async<fp64_t>(const Tile<fp64_t> &A);

// Explicit instantiation
template
void drelu<fp32_t>(const Tile<fp32_t> &A);

template
void drelu<fp64_t>(const Tile<fp64_t> &A);

} // namespace tile
} // namespace nntile

