/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/multiply_fiber.cc
 * Logical graph multiply fiber operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/multiply_fiber.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Multiply along fibers: output = alpha * fiber * tensor
LogicalGraph::TensorNode& multiply_fiber(
    Scalar alpha,
    LogicalGraph::TensorNode& fiber,
    LogicalGraph::TensorNode& tensor,
    const std::string& output_name,
    Index axis,
    Index batch_ndim)
{
    if(&fiber.graph() != &tensor.graph())
    {
        throw std::invalid_argument(
            "multiply_fiber: tensors must belong to the same graph");
    }

    if(fiber.dtype() != tensor.dtype())
    {
        throw std::invalid_argument(
            "multiply_fiber: all tensors must have the same dtype");
    }

    if(axis < 0 || axis >= fiber.ndim())
    {
        throw std::invalid_argument(
            "multiply_fiber: axis out of bounds");
    }

    if(batch_ndim < 0 || axis + batch_ndim > fiber.ndim())
    {
        throw std::invalid_argument(
            "multiply_fiber: invalid batch_ndim");
    }

    // Check fiber dimensions (should be 1D for the axis being broadcast)
    if(fiber.ndim() != 1)
    {
        throw std::invalid_argument(
            "multiply_fiber: fiber tensor must be 1-dimensional");
    }

    // Check that fiber size matches the dimension being broadcast
    if(fiber.shape()[0] != tensor.shape()[axis])
    {
        throw std::invalid_argument(
            "multiply_fiber: fiber size must match tensor dimension at specified axis");
    }

    // Output has the same shape as the input tensor
    std::vector<Index> output_shape = tensor.shape();
    LogicalGraph::TensorNode& output = fiber.graph().tensor(
        std::move(output_shape),
        output_name,
        fiber.dtype());

    OpAttrs attrs = MultiplyFiberAttrs{axis, batch_ndim, alpha};
    fiber.graph().add_op(
        OpType::MULTIPLY_FIBER,
        attrs,
        {&fiber, &tensor},
        {&output}
    );

    return output;
}

} // namespace nntile::graph
