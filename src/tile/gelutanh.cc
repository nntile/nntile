/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/gelu.cc
 * Approximate GeLU operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-27
 * */

#include "nntile/tile/gelutanh.hh"
#include "nntile/starpu/gelutanh.hh"

namespace nntile
{
namespace tile
{

//! Blocking version of tile-wise approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise GeLU operation
 * */
template<typename T>
void gelutanh_async(const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::gelutanh::submit<T>(A.nelems, A);
}

//! Blocking version of tile-wise approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise GeLU operation
 * */
template<typename T>
void gelutanh(const Tile<T> &A)
{
    gelutanh_async<T>(A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void gelutanh<fp32_t>(const Tile<fp32_t> &A);

template
void gelutanh<fp64_t>(const Tile<fp64_t> &A);

} // namespace tile
} // namespace nntile

