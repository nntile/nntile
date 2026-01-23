/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/multiply_slice.cc
 * Logical graph multiply slice operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical_graph_ops.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Multiply slice: tensor = alpha * slice * tensor
void multiply_slice(
    Scalar alpha,
    LogicalGraph::TensorNode& slice,
    LogicalGraph::TensorNode& tensor,
    Index axis)
{
    if(&slice.graph() != &tensor.graph())
    {
        throw std::invalid_argument(
            "multiply_slice: tensors must belong to the same graph");
    }

    if(slice.dtype() != tensor.dtype())
    {
        throw std::invalid_argument(
            "multiply_slice: all tensors must have the same dtype");
    }

    if(axis < 0 || axis >= slice.ndim())
    {
        throw std::invalid_argument(
            "multiply_slice: axis out of bounds");
    }

    // Check that shapes are compatible for slice-wise operation
    if(slice.shape() != tensor.shape())
    {
        throw std::invalid_argument(
            "multiply_slice: tensors must have the same shape");
    }

    OpAttrs attrs = MultiplySliceAttrs{axis, alpha, 0.0};  // beta is not used in this operation
    slice.graph().add_op(
        OpType::MULTIPLY_SLICE,
        attrs,
        {&slice, &tensor},
        {&tensor}
    );
}

} // namespace nntile::graph