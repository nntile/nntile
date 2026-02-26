/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/logsumexp.cc
 * Logical graph logsumexp operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/logsumexp.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Log sum exp from maxsumexp output: y = max + log(sumexp)
void logsumexp(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "logsumexp: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "logsumexp: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "logsumexp: axis out of bounds");
    }

    OpAttrs attrs = LogSumExpAttrs{1.0, 0.0, axis};  // alpha=1, beta=0
    x.graph().add_op(
        OpType::LOGSUMEXP,
        attrs,
        {&x},
        {&y}
    );
}

} // namespace nntile::graph
