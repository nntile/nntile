/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/scale_fiber.cc
 * Logical graph scale_fiber operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/scale_fiber.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Scale along fibers: y = alpha * scale_fiber(x, y)
void scale_fiber(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Index axis,
    Index batch_ndim)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "scale_fiber: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "scale_fiber: tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "scale_fiber: axis out of bounds");
    }

    if(batch_ndim < 0 || axis + batch_ndim > x.ndim())
    {
        throw std::invalid_argument(
            "scale_fiber: invalid batch_ndim");
    }

    OpAttrs attrs = ReductionAttrs{alpha, 0.0, axis, batch_ndim, 0};  // beta=0, redux=0
    x.graph().add_op(
        OpType::SCALE_FIBER,
        attrs,
        {&x, &y},
        {&y}
    );
}

} // namespace nntile::graph