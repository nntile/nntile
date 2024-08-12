/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/scal_inplace.cc
 * Inplace scal of Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/scal_inplace.hh"
#include "nntile/starpu/scal_inplace.hh"

namespace nntile::tile
{

//! Tile-wise scal_inplaceing
template<typename T>
void scal_inplace_async(Scalar alpha, const Tile<T> &data)
{
    // Insert task
    starpu::scal_inplace::submit<T>(data.nelems, alpha, data);
}

//! Tile-wise scal_inplaceing
template<typename T>
void scal_inplace(Scalar alpha, const Tile<T> &data)
{
    scal_inplace_async<T>(alpha, data);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void scal_inplace_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &data);

template
void scal_inplace_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &data);

template
void scal_inplace_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &data);

// Explicit instantiation
template
void scal_inplace<fp32_t>(Scalar alpha, const Tile<fp32_t> &data);

template
void scal_inplace<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &data);

template
void scal_inplace<fp64_t>(Scalar alpha, const Tile<fp64_t> &data);

} // namespace nntile::tile
