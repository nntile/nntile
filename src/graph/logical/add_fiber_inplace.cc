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

//! Add along fibers in-place: y = alpha * x + beta * y
void add_fiber_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "add_fiber_inplace: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "add_fiber_inplace: all tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "add_fiber_inplace: axis out of bounds");
    }

    // Check that shapes are compatible for fiber-wise operation
    if(x.shape() != y.shape())
    {
        throw std::invalid_argument(
            "add_fiber_inplace: tensors must have the same shape");
    }

    OpAttrs attrs = AddFiberAttrs{alpha};
    x.graph().add_op(
        OpType::ADD_FIBER_INPLACE,
        attrs,
        {&x, &y},
        {&y}
    );
}

} // namespace nntile::graph