/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/transpose.cc
 * Logical graph transpose operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/transpose.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Transpose operation: y = alpha * transpose(x)
LogicalGraph::TensorNode& transpose(
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    Scalar alpha,
    Index ndim)
{
    // For transpose, we need to reverse the first ndim dimensions
    std::vector<Index> output_shape = x.shape();
    if(ndim > 0 && ndim <= x.ndim())
    {
        // Reverse the first ndim dimensions
        for(Index i = 0; i < ndim/2; ++i)
        {
            std::swap(output_shape[i], output_shape[ndim-1-i]);
        }
    }

    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = TransposeAttrs{alpha, ndim};
    x.graph().add_op(
        OpType::TRANSPOSE,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

} // namespace nntile::graph