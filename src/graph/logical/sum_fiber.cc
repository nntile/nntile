/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/sum_fiber.cc
 * Logical graph sum_fiber operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/sum_fiber.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Sum along fibers: y = alpha * sum_fiber(x) + beta * y
void sum_fiber(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "sum_fiber: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "sum_fiber: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "sum_fiber: axis out of bounds");
    }

    if(batch_ndim < 0 || axis + batch_ndim > x.ndim())
    {
        throw std::invalid_argument(
            "sum_fiber: invalid batch_ndim");
    }

    auto attrs = std::make_shared<ReductionAttrs>(ReductionAttrs{alpha, beta, axis, batch_ndim, redux});
    x.graph().add_op(
        OpType::SUM_FIBER,
        attrs,
        {&x, &y},
        {&y}
    );
}

} // namespace nntile::graph
