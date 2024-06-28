/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/fill.cc
 * Fill operation for Tile<T>
 *
 * @version 1.0.0
 * */

#include "nntile/tile/fill.hh"
#include "nntile/starpu/fill.hh"

namespace nntile::tile
{

//! Asynchronous tile-wise fill operation
/*! @param[inout] A: Tile for the element-wise fill operation
 * */
template<typename T>
void fill_async(scal_t val, const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::fill::submit<T>(A.nelems, val, A);
}

//! Blocking version of tile-wise flll operation
/*! @param[inout] A: Tile for the element-wise fill operation
 * */
template<typename T>
void fill(scal_t val, const Tile<T> &A)
{
    fill_async<T>(val, A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void fill_async<fp32_t>(scal_t val, const Tile<fp32_t> &A);

template
void fill_async<fp32_fast_tf32_t>(scal_t val, const Tile<fp32_fast_tf32_t> &A);

template
void fill_async<fp64_t>(scal_t val, const Tile<fp64_t> &A);

// Explicit instantiation
template
void fill<fp32_t>(scal_t val, const Tile<fp32_t> &A);

template
void fill<fp32_fast_tf32_t>(scal_t val, const Tile<fp32_fast_tf32_t> &A);

template
void fill<fp64_t>(scal_t val, const Tile<fp64_t> &A);

} // namespace nntile::tile
