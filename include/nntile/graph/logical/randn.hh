/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/randn.hh
 * Logical graph randn operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>
#include <vector>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Random normal generation: x = randn(start, underlying_shape, seed, mean, stddev)
//! @param x Output tensor (modified in-place)
//! @param start Starting indices for the random region
//! @param underlying_shape Shape of the underlying tensor
//! @param seed Random seed
//! @param mean Mean of the normal distribution
//! @param stddev Standard deviation of the normal distribution
void randn(
    LogicalGraph::TensorNode& x,
    const std::vector<Index>& start,
    const std::vector<Index>& underlying_shape,
    unsigned long long seed = 0,
    Scalar mean = 0.0,
    Scalar stddev = 1.0
);

} // namespace nntile::graph