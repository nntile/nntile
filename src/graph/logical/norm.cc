/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/norm.cc
 * Logical graph norm operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/norm.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Euclidean norm: y = alpha * norm(x) + beta * y
void norm(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "norm: tensors must belong to the same graph");
    }

    if(y.ndim() != 1 || y.shape()[0] != 1)
    {
        throw std::invalid_argument(
            "norm: output tensor must be scalar (shape [1])");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "norm: input and output tensors must have the same dtype");
    }

    OpAttrs attrs = TotalSumAttrs{alpha, beta};
    x.graph().add_op(
        OpType::NORM,
        attrs,
        {&x, &y},
        {&y}
    );
}

} // namespace nntile::graph