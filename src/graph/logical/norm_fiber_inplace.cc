/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/norm_fiber_inplace.cc
 * Logical graph norm_fiber_inplace operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/norm_fiber_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Norm along fibers (in-place): y = alpha * norm_fiber(x) + beta * y
void norm_fiber_inplace(
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
            "norm_fiber_inplace: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "norm_fiber_inplace: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "norm_fiber_inplace: axis out of bounds");
    }

    if(batch_ndim < 0 || axis + batch_ndim > x.ndim())
    {
        throw std::invalid_argument(
            "norm_fiber_inplace: invalid batch_ndim");
    }

    OpAttrs attrs = ReductionAttrs{alpha, beta, axis, batch_ndim, redux};
    x.graph().add_op(
        OpType::NORM_FIBER_INPLACE,
        attrs,
        {&x, &y},
        {&y}
    );
}

} // namespace nntile::graph
