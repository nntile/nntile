/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/sum.hh
 * Sum all elements of a Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core.hh>

namespace nntile::core
{

//! Tile-wise sum
template<typename T>
void sum_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst);

//! Tile-wise sum
template<typename T>
void sum(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst);

} // namespace nntile::core
