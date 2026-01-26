/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/multiply_inplace.cc
 * Logical graph multiply_inplace operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/multiply_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Multiply in-place: y = x * y
void multiply_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y)
{
    // Validate inputs belong to the same graph
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must belong to the same graph");
    }

    // Validate input dtypes match
    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must have the same dtype");
    }

    // Validate shapes match
    if(x.shape() != y.shape())
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must have the same shape");
    }

    OpAttrs attrs = BinaryOpAttrs{1.0, 1.0};
    x.graph().add_op(
        OpType::MULTIPLY_INPLACE,
        attrs,
        {&x, &y},
        {&y}
    );
}

} // namespace nntile::graph