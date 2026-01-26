/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/add_slice.cc
 * Logical graph add slice operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/add_slice.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Add along slices: output = alpha * slice + beta * tensor
LogicalGraph::TensorNode& add_slice(
    Scalar alpha,
    LogicalGraph::TensorNode& slice,
    Scalar beta,
    LogicalGraph::TensorNode& tensor,
    const std::string& output_name,
    Index axis)
{
    if(&slice.graph() != &tensor.graph())
    {
        throw std::invalid_argument(
            "add_slice: tensors must belong to the same graph");
    }

    if(slice.dtype() != tensor.dtype())
    {
        throw std::invalid_argument(
            "add_slice: all tensors must have the same dtype");
    }

    if(axis < 0 || axis >= tensor.ndim())
    {
        throw std::invalid_argument(
            "add_slice: axis out of bounds");
    }

    // Check basic dimension compatibility
    if(slice.ndim() + 1 != tensor.ndim())
    {
        throw std::invalid_argument(
            "add_slice: slice must have one fewer dimension than tensor");
    }

    // Check that slice shape is compatible with tensor shape for broadcasting
    Index slice_dim = 0;
    for(Index i = 0; i < tensor.ndim(); ++i)
    {
        if(i != axis)
        {
            if(slice_dim >= slice.ndim() ||
               slice.shape()[slice_dim] != tensor.shape()[i])
            {
                throw std::invalid_argument(
                    "add_slice: slice shape incompatible with tensor shape");
            }
            ++slice_dim;
        }
    }

    // Output has the same shape as the input tensor
    std::vector<Index> output_shape = tensor.shape();
    LogicalGraph::TensorNode& output = slice.graph().tensor(
        std::move(output_shape),
        output_name,
        slice.dtype());

    OpAttrs attrs = AddSliceAttrs{axis, alpha, beta};
    slice.graph().add_op(
        OpType::ADD_SLICE,
        attrs,
        {&slice, &tensor},
        {&output}
    );

    return output;
}

} // namespace nntile::graph