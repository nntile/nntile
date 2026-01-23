/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/add_inplace.cc
 * Logical graph add_inplace operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/add_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Add in-place: y = alpha * x + beta * y
void add_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Scalar beta)
{
    // Validate inputs belong to the same graph
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "add_inplace: input tensors must belong to the same graph");
    }

    // Validate input dtypes match
    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "add_inplace: input tensors must have the same dtype");
    }

    // Validate shapes match
    if(x.shape() != y.shape())
    {
        throw std::invalid_argument(
            "add_inplace: input tensors must have the same shape");
    }

    OpAttrs attrs = BinaryOpAttrs{alpha, beta};
    x.graph().add_op(
        OpType::ADD_INPLACE,
        attrs,
        {&x, &y},
        {&y}
    );
}

} // namespace nntile::graph