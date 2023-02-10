/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/add_scalar.cc
 * Add scalar to elements from Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#include "nntile/tile/add_scalar.hh"
#include "nntile/starpu/add_scalar.hh"

namespace nntile
{
namespace tile
{

//! Asynchronous version of tile-wise add scalar operation
/*! @param[in] val: Input scalar value
 * @param[inout] src: Input and output tile for the add_scalar operation
 * */
template<typename T>
void add_scalar_async(T val, const Tile<T> &src)
{
    // Submit task
    starpu::add_scalar::submit<T>(val, src.nelems, src);
}

//! Blocking version of tile-wise add_scalar operation
/*! @param[in] val: Input scalar value
 * @param[inout] src: Input and output tile for the add_scalar operation
 * */
template<typename T>
void add_scalar(T val, const Tile<T> &src)
{
    add_scalar_async<T>(val, src);
    starpu_task_wait_for_all();
}


// Explicit instantiation
template
void add_scalar_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src);

template
void add_scalar_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src);

// Explicit instantiation
template
void add_scalar<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src);

template
void add_scalar<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src);

} // namespace tile
} // namespace nntile
