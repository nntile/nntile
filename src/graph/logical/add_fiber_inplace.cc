/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/add_fiber_inplace.cc
 * Logical graph add fiber in-place operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/add_fiber_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Add along fibers in-place: tensor = alpha * fiber + beta * tensor
void add_fiber_inplace(
    Scalar alpha,
    LogicalGraph::TensorNode& fiber,
    Scalar beta,
    LogicalGraph::TensorNode& tensor,
    Index axis,
    Index batch_ndim)
{
    if(&fiber.graph() != &tensor.graph())
    {
        throw std::invalid_argument(
            "add_fiber_inplace: tensors must belong to the same graph");
    }

    if(fiber.dtype() != tensor.dtype())
    {
        throw std::invalid_argument(
            "add_fiber_inplace: all tensors must have the same dtype");
    }

    if(axis < 0 || axis >= tensor.ndim() - batch_ndim)
    {
        throw std::invalid_argument(
            "add_fiber_inplace: axis out of bounds");
    }

    if(batch_ndim < 0 || axis + batch_ndim > fiber.ndim())
    {
        throw std::invalid_argument(
            "add_fiber_inplace: invalid batch_ndim");
    }

    // Check fiber dimensions (should have batch_ndim+1 dimensions)
    if(fiber.ndim() != batch_ndim + 1)
    {
        throw std::invalid_argument(
            "add_fiber_inplace: fiber tensor must have batch_ndim+1 dimensions");
    }

    // Check that fiber size matches the dimension being broadcast
    if(fiber.shape()[0] != tensor.shape()[axis])
    {
        throw std::invalid_argument(
            "add_fiber_inplace: fiber size must match tensor dimension at specified axis");
    }

    // Check batch dimensions compatibility
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(fiber.shape()[i+1] != tensor.shape()[tensor.ndim() - batch_ndim + i])
        {
            throw std::invalid_argument(
                "add_fiber_inplace: fiber and tensor batch dimensions must match");
        }
    }

    auto attrs = std::make_shared<AddFiberAttrs>(AddFiberAttrs{axis, batch_ndim, alpha, beta});
    fiber.graph().add_op(
        OpType::ADD_FIBER_INPLACE,
        attrs,
        {&fiber, &tensor},
        {&tensor}
    );
}

} // namespace nntile::graph
