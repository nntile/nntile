/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/maxsumexp.cc
 * Logical graph maxsumexp operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/maxsumexp.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Max and sum of exponents along axis:
//! y[0, ...] = max(x)
//! y[1, ...] = sum(exp(x - y[0, ...]))
void maxsumexp(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    int redux)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "maxsumexp: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "maxsumexp: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "maxsumexp: axis out of bounds");
    }

    OpAttrs attrs = LogSumExpAttrs{1.0, 0.0, axis};  // alpha=1, beta=0, axis
    x.graph().add_op(
        OpType::MAXSUMEXP,
        attrs,
        {&x},
        {&y}
    );
}

} // namespace nntile::graph
