/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/multiply_fiber_inplace.cc
 * Logical graph multiply fiber in-place operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/multiply_fiber_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Multiply along fibers in-place: tensor = alpha * fiber * tensor
void multiply_fiber_inplace(
    Scalar alpha,
    LogicalGraph::TensorNode& fiber,
    LogicalGraph::TensorNode& tensor,
    Index axis,
    Index batch_ndim)
{
    if(&fiber.graph() != &tensor.graph())
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: tensors must belong to the same graph");
    }

    if(fiber.dtype() != tensor.dtype())
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: all tensors must have the same dtype");
    }

    if(axis < 0 || axis >= fiber.ndim())
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: axis out of bounds");
    }

    if(batch_ndim < 0 || axis + batch_ndim > fiber.ndim())
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: invalid batch_ndim");
    }

    // Check fiber dimensions (should be 1D for the axis being broadcast)
    if(fiber.ndim() != 1)
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: fiber tensor must be 1-dimensional");
    }

    // Check that fiber size matches the dimension being broadcast
    if(fiber.shape()[0] != tensor.shape()[axis])
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: fiber size must match tensor dimension at specified axis");
    }

    auto attrs = std::make_shared<MultiplyFiberAttrs>(MultiplyFiberAttrs{axis, batch_ndim, alpha});
    fiber.graph().add_op(
        OpType::MULTIPLY_FIBER_INPLACE,
        attrs,
        {&fiber, &tensor},
        {&tensor}
    );
}

} // namespace nntile::graph
