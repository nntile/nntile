/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/scale_slice.cc
 * Logical graph scale_slice operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/scale_slice.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Scale along slices: y = alpha * scale_slice(x, y)
void scale_slice(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Index axis)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "scale_slice: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "scale_slice: tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "scale_slice: axis out of bounds");
    }

    OpAttrs attrs = ReductionAttrs{alpha, 0.0, axis, 0, 0};  // batch_ndim=0, redux=0
    x.graph().add_op(
        OpType::SCALE_SLICE,
        attrs,
        {&x, &y},
        {&y}
    );
}

} // namespace nntile::graph