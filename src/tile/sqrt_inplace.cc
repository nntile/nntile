/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sqrt_inplace.cc
 * Inplace sqrt operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/sqrt_inplace.hh"
#include "nntile/starpu/sqrt_inplace.hh"

namespace nntile::tile
{

//! Asynchronous tile-wise sqrt operation
/*! @param[inout] A: Tile for the element-wise sqrt operation
 * */
template<typename T>
void sqrt_inplace_async(const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::sqrt_inplace::submit<T>(A.nelems, A);
}

//! Blocking version of tile-wise sqrt operation
/*! @param[inout] A: Tile for the element-wise sqrt operation
 * */
template<typename T>
void sqrt_inplace(const Tile<T> &A)
{
    sqrt_inplace_async<T>(A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void sqrt_inplace_async<fp32_t>(const Tile<fp32_t> &A);

template
void sqrt_inplace_async<fp64_t>(const Tile<fp64_t> &A);

// Explicit instantiation
template
void sqrt_inplace<fp32_t>(const Tile<fp32_t> &A);

template
void sqrt_inplace<fp64_t>(const Tile<fp64_t> &A);

} // namespace nntile::tile
