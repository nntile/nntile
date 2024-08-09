/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/add_scalar.cc
 * Add_scalar operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/add_scalar.hh"
#include "nntile/starpu/add_scalar.hh"

namespace nntile::tile
{

//! Tile-wise add_scalar operation
template<typename T>
void add_scalar_async(Scalar alpha, Scalar beta, const Tile<T> &dst)
{
    // Do nothing if alpha is zero and beta is one
    if(alpha == 0.0 && beta == 1.0)
    {
        return;
    }
    // Insert corresponding task
    starpu::add_scalar::submit<T>(dst.nelems, alpha, beta, dst);
}

//! Tile-wise add_scalar operation
template<typename T>
void add_scalar(Scalar alpha, Scalar beta, const Tile<T> &dst)
{
    add_scalar_async<T>(alpha, beta, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void add_scalar_async<fp32_t>(Scalar alpha, Scalar beta, const Tile<fp32_t> &dst);

template
void add_scalar_async<fp64_t>(Scalar alpha, Scalar beta, const Tile<fp64_t> &dst);

// Explicit instantiation of template
template
void add_scalar<fp32_t>(Scalar alpha, Scalar beta, const Tile<fp32_t> &dst);

template
void add_scalar<fp64_t>(Scalar alpha, Scalar beta, const Tile<fp64_t> &dst);

} // namespace nntile::tile
