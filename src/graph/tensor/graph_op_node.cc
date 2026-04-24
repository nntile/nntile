/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/graph_op_node.cc
 * TensorGraph::OpNode default member implementations.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/graph.hh"

#include <stdexcept>
#include <string>

namespace nntile::graph
{

struct LoweringContext;

void TensorGraph::OpNode::lower_to_tile(const LoweringContext&) const
{
    throw std::runtime_error(
        "lower_to_tile is not implemented for tensor op '" + op_name() + "'");
}

} // namespace nntile::graph
