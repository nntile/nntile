/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/sqrt.cc
 * Logical graph sqrt operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/sqrt.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Sqrt activation: y = sqrt(x)
LogicalGraph::TensorNode& sqrt(
    LogicalGraph::TensorNode& x,
    const std::string& output_name)
{
    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    auto attrs = std::make_shared<GeluAttrs>(GeluAttrs{});
    x.graph().add_op(
        OpType::SQRT,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

} // namespace nntile::graph
