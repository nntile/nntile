/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/sum.cc
 * Logical graph sum operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/sum.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Total sum of all elements: y = alpha * sum(x) + beta * y
void sum(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "sum: tensors must belong to the same graph");
    }

    if(y.ndim() != 0)
    {
        throw std::invalid_argument(
            "sum: output tensor must be scalar (shape [])");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "sum: input and output tensors must have the same dtype");
    }

    OpAttrs attrs = TotalSumAttrs{alpha, beta};
    x.graph().add_op(
        OpType::SUM,
        attrs,
        {&x, &y},
        {&y}
    );
}

} // namespace nntile::graph
