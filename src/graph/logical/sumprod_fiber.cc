/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/sumprod_fiber.cc
 * Logical graph sumprod_fiber operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/sumprod_fiber.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Sum of products along fibers: y = alpha * sum_fiber(x1 * x2) + beta * y
void sumprod_fiber(
    LogicalGraph::TensorNode& x1,
    LogicalGraph::TensorNode& x2,
    LogicalGraph::TensorNode& y,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(&x1.graph() != &x2.graph() || &x1.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "sumprod_fiber: tensors must belong to the same graph");
    }

    if(x1.dtype() != x2.dtype() || x1.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "sumprod_fiber: all tensors must have the same dtype");
    }

    if(x1.shape() != x2.shape())
    {
        throw std::invalid_argument(
            "sumprod_fiber: x1 and x2 must have the same shape");
    }

    if(axis < 0 || axis >= x1.ndim())
    {
        throw std::invalid_argument(
            "sumprod_fiber: axis out of bounds");
    }

    auto attrs = std::make_shared<ReductionAttrs>(ReductionAttrs{alpha, beta, axis, 0, redux});
    x1.graph().add_op(
        OpType::SUMPROD_FIBER,
        attrs,
        {&x1, &x2, &y},
        {&y}
    );
}

} // namespace nntile::graph
