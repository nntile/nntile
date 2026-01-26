/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/clear.cc
 * Logical graph clear operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/clear.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Clear tensor: x = 0
void clear(LogicalGraph::TensorNode& x)
{
    OpAttrs attrs = ClearAttrs{};

    // In-place operation: inputs and outputs are the same tensor
    x.graph().add_op(
        OpType::CLEAR,
        attrs,
        {},
        {&x}
    );
}

} // namespace nntile::graph