/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/softmax_inplace.cc
 * Logical graph softmax in-place operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/softmax_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Softmax in-place: y = softmax(maxsumexp, y, alpha)
void softmax_inplace(
    LogicalGraph::TensorNode& maxsumexp,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Index axis)
{
    if(&maxsumexp.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "softmax_inplace: tensors must belong to the same graph");
    }

    if(maxsumexp.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "softmax_inplace: tensors must have the same dtype");
    }

    if(axis < 0)
    {
        axis += y.ndim();
    }

    if(axis < 0 || axis >= y.ndim())
    {
        throw std::invalid_argument(
            "softmax_inplace: axis out of bounds");
    }

    OpAttrs attrs = LogSumExpAttrs{alpha, 1.0, axis};
    y.graph().add_op(
        OpType::SOFTMAX_INPLACE,
        attrs,
        {&maxsumexp, &y},
        {&y}
    );
}

} // namespace nntile::graph
