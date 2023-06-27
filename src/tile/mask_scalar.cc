/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/mask_scalar.cc
 * Mask scalar operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-22
 * */

#include "nntile/tile/mask_scalar.hh"
#include "nntile/starpu/mask_scalar.hh"

namespace nntile
{
namespace tile
{

//! Asynchronous tile-wise mask scalar operation
/*! @param[inout] A: Tile for the element-wise mask scalar operation
 * */
template<typename T>
void mask_scalar_async(const Tile<bool_t> &mask, T val, const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::mask_scalar::submit<T>(A.nelems, A.shape[2], mask, val, A);
}

//! Blocking version of tile-wise mask scalar operation
/*! @param[inout] A: Tile for the element-wise mask scalar operation
 * */
template<typename T>
void mask_scalar(const Tile<bool_t> &mask, T val, const Tile<T> &A)
{
    mask_scalar_async<T>(mask, val, A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void mask_scalar_async<fp32_t>(const Tile<bool_t> &mask, fp32_t val, const Tile<fp32_t> &A);

template
void mask_scalar_async<fp64_t>(const Tile<bool_t> &mask, fp64_t val, const Tile<fp64_t> &A);

// Explicit instantiation
template
void mask_scalar<fp32_t>(const Tile<bool_t> &mask, fp32_t val, const Tile<fp32_t> &A);

template
void mask_scalar<fp64_t>(const Tile<bool_t> &mask, fp64_t val, const Tile<fp64_t> &A);

} // namespace tile
} // namespace nntile