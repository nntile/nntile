/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/gelutanh_inplace.cc
 * Approximate GeLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/gelutanh_inplace.hh"
#include "nntile/starpu/gelutanh_inplace.hh"

namespace nntile::tile
{

//! Asyncrhonous tile-wise approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise GeLU operation
 * */
template<typename T>
void gelutanh_inplace_async(const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::gelutanh_inplace::submit<T>(A.nelems, A);
}

//! Blocking version of tile-wise approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise GeLU operation
 * */
template<typename T>
void gelutanh_inplace(const Tile<T> &A)
{
    gelutanh_inplace_async<T>(A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void gelutanh_inplace_async<fp32_t>(const Tile<fp32_t> &A);

template
void gelutanh_inplace_async<fp64_t>(const Tile<fp64_t> &A);

// Explicit instantiation
template
void gelutanh_inplace<fp32_t>(const Tile<fp32_t> &A);

template
void gelutanh_inplace<fp64_t>(const Tile<fp64_t> &A);

} // namespace nntile::tile
