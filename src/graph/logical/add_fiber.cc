/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/add_fiber.cc
 * Logical graph add fiber operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/add_fiber.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Add along fibers: output = alpha * fiber + beta * tensor
LogicalGraph::TensorNode& add_fiber(
    LogicalGraph::TensorNode& fiber,
    LogicalGraph::TensorNode& tensor,
    const std::string& output_name,
    Index axis,
    Scalar alpha,
    Scalar beta)
{
    if(&fiber.graph() != &tensor.graph())
    {
        throw std::invalid_argument(
            "add_fiber: tensors must belong to the same graph");
    }

    if(fiber.dtype() != tensor.dtype())
    {
        throw std::invalid_argument(
            "add_fiber: all tensors must have the same dtype");
    }

    if(axis < 0 || axis >= tensor.ndim())
    {
        throw std::invalid_argument(
            "add_fiber: axis out of bounds");
    }

    // Check fiber dimensions (should be 1D for the axis being broadcast)
    if(fiber.ndim() != 1)
    {
        throw std::invalid_argument(
            "add_fiber: fiber tensor must be 1-dimensional");
    }

    // Check that fiber size matches the dimension being broadcast
    if(fiber.shape()[0] != tensor.shape()[axis])
    {
        throw std::invalid_argument(
            "add_fiber: fiber size must match tensor dimension at specified axis");
    }

    // Output has the same shape as the input tensor
    std::vector<Index> output_shape = tensor.shape();
    LogicalGraph::TensorNode& output = fiber.graph().tensor(
        std::move(output_shape),
        output_name,
        fiber.dtype());

    OpAttrs attrs = AddFiberAttrs{alpha};
    fiber.graph().add_op(
        OpType::ADD_FIBER,
        attrs,
        {&fiber, &tensor},
        {&output}
    );

    return output;
}

} // namespace nntile::graph