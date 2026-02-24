/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/norm_slice.cc
 * Logical graph norm_slice operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/norm_slice.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Norm along slices (out-of-place): dst = alpha * norm_slice(x) + beta * src2
void norm_slice(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& src2,
    LogicalGraph::TensorNode& dst,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &src2.graph() || &x.graph() != &dst.graph())
    {
        throw std::invalid_argument(
            "norm_slice: tensors must belong to the same graph");
    }

    if(&src2 == &dst)
    {
        throw std::invalid_argument(
            "norm_slice: use norm_slice_inplace when src2 and dst are the same");
    }

    if(x.dtype() != src2.dtype() || x.dtype() != dst.dtype())
    {
        throw std::invalid_argument(
            "norm_slice: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "norm_slice: axis out of bounds");
    }

    OpAttrs attrs = ReductionAttrs{alpha, beta, axis, 0, redux};  // batch_ndim = 0 for slice
    x.graph().add_op(
        OpType::NORM_SLICE,
        attrs,
        {&x, &src2},
        {&dst}
    );
}

} // namespace nntile::graph
