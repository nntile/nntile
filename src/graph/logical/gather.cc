/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/gather.cc
 * Logical graph gather operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/gather.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Gather operation: y = gather(x)
LogicalGraph::TensorNode& gather(
    LogicalGraph::TensorNode& x,
    const std::string& output_name)
{
    // For now, assume gather doesn't change shape
    // In practice, this would depend on the indices
    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    x.graph().add_op(
        OpType::GATHER,
        nullptr,
        {&x},
        {&output}
    );

    return output;
}

} // namespace nntile::graph
