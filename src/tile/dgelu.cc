/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/dgelu.cc
 * Derivative of GeLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/dgelu.hh"
#include "nntile/starpu/dgelu.hh"

namespace nntile::tile
{

//! Blocking version of tile-wise derivative of GeLU operation
/*! @param[inout] A: Tile for the element-wise derivative of GeLU operation
 * */
template<typename T>
void dgelu_async(const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::dgelu::submit<T>(A.nelems, A);
}

//! Blocking version of tile-wise derivative of GeLU operation
/*! @param[inout] A: Tile for the element-wise derivative of GeLU operation
 * */
template<typename T>
void dgelu(const Tile<T> &A)
{
    dgelu_async<T>(A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void dgelu_async<fp32_t>(const Tile<fp32_t> &A);

template
void dgelu_async<fp64_t>(const Tile<fp64_t> &A);

// Explicit instantiation
template
void dgelu<fp32_t>(const Tile<fp32_t> &A);

template
void dgelu<fp64_t>(const Tile<fp64_t> &A);

} // namespace nntile::tile
