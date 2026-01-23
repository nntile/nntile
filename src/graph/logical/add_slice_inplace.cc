/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/add_slice_inplace.cc
 * Logical graph add slice in-place operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/add_slice_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Add along slices in-place: tensor = alpha * slice + beta * tensor
void add_slice_inplace(
    Scalar alpha,
    LogicalGraph::TensorNode& slice,
    Scalar beta,
    LogicalGraph::TensorNode& tensor,
    Index axis)
{
    if(&slice.graph() != &tensor.graph())
    {
        throw std::invalid_argument(
            "add_slice_inplace: tensors must belong to the same graph");
    }

    if(slice.dtype() != tensor.dtype())
    {
        throw std::invalid_argument(
            "add_slice_inplace: all tensors must have the same dtype");
    }

    if(axis < 0 || axis >= slice.ndim())
    {
        throw std::invalid_argument(
            "add_slice_inplace: axis out of bounds");
    }

    // Check that shapes are compatible for slice-wise operation
    if(slice.shape() != tensor.shape())
    {
        throw std::invalid_argument(
            "add_slice_inplace: tensors must have the same shape");
    }

    OpAttrs attrs = AddSliceAttrs{axis, alpha, beta};
    slice.graph().add_op(
        OpType::ADD_SLICE_INPLACE,
        attrs,
        {&slice, &tensor},
        {&tensor}
    );
}

} // namespace nntile::graph