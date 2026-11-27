/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/log_scalar.hh
 * Log scalar value from Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>
#include <string>

namespace nntile::core
{

template<typename T>
void log_scalar_async(int starpu_worker_hint, const std::string &name, const Tile<T> &value);

template<typename T>
void log_scalar(int starpu_worker_hint, const std::string &name, const Tile<T> &value);

} // namespace nntile::core
