/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/hypot.cc
 * Logical graph hypot operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/hypot.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Hypot operation: z = hypot(alpha * x, beta * y)
LogicalGraph::TensorNode& hypot(
    Scalar alpha,
    LogicalGraph::TensorNode& x,
    Scalar beta,
    LogicalGraph::TensorNode& y,
    const std::string& output_name)
{
    // Validate inputs belong to the same graph
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "hypot: input tensors must belong to the same graph");
    }

    // Validate input dtypes match
    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "hypot: input tensors must have the same dtype");
    }

    // Validate shapes match
    if(x.shape() != y.shape())
    {
        throw std::invalid_argument(
            "hypot: input tensors must have the same shape");
    }

    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = BinaryOpAttrs{alpha, beta};
    x.graph().add_op(
        OpType::HYPOT,
        attrs,
        {&x, &y},
        {&output}
    );

    return output;
}

} // namespace nntile::graph
