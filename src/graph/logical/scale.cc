/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/scale.cc
 * Logical graph scale operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/scale.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Scale operation: y = alpha * x
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scaling factor (default: 1.0)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& scale(
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    Scalar alpha)
{
    // Create output tensor with same shape and dtype as input
    LogicalGraph::TensorNode& output = x.graph().tensor(
        x.shape(),
        output_name,
        x.dtype());

    // Create operation attributes
    auto attrs = std::make_shared<ScaleAttrs>(ScaleAttrs{alpha});

    // Add operation to graph
    x.graph().add_op(
        OpType::SCALE,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

} // namespace nntile::graph
