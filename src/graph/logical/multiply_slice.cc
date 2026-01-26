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

//! Multiply slice: tensor = alpha * tensor * slice (broadcasted along axis)
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

    // Check dimension compatibility: tensor should have one more dimension than slice
    if(tensor.ndim() != slice.ndim() + 1)
    {
        throw std::invalid_argument(
            "multiply_slice: tensor.ndim() must equal slice.ndim() + 1");
    }

    if(axis < 0 || axis >= tensor.ndim())
    {
        throw std::invalid_argument(
            "multiply_slice: axis out of bounds for tensor");
    }

    // Check that shapes are compatible for broadcasting
    // Dimensions before axis and after axis (adjusted for slice) must match
    for(Index i = 0; i < axis; ++i)
    {
        if(slice.shape()[i] != tensor.shape()[i])
        {
            throw std::invalid_argument(
                "multiply_slice: shape mismatch before broadcast axis");
        }
    }
    for(Index i = axis; i < slice.ndim(); ++i)
    {
        if(slice.shape()[i] != tensor.shape()[i+1])
        {
            throw std::invalid_argument(
                "multiply_slice: shape mismatch after broadcast axis");
        }
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