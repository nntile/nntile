/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/randn.cc
 * Logical graph randn operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/randn.hh"

// Include standard headers
#include <stdexcept>
#include <utility>
#include <vector>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Random normal generation: x = randn(start, underlying_shape, seed, mean, stddev)
void randn(
    LogicalGraph::TensorNode& x,
    const std::vector<Index>& start,
    const std::vector<Index>& underlying_shape,
    unsigned long long seed,
    Scalar mean,
    Scalar stddev)
{
    auto attrs = std::make_shared<RandnAttrs>(RandnAttrs{start, underlying_shape, seed, mean, stddev});
    x.graph().add_op(
        OpType::RANDN,
        attrs,
        {},
        {&x}
    );
}

} // namespace nntile::graph
