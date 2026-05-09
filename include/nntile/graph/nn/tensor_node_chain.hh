/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/tensor_node_chain.hh
 * Inline fluent helpers on ``NNGraph::TensorNode`` (include after
 * ``graph_ops``).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/nn/ops/add.hh>

namespace nntile::graph
{

inline NNGraph::TensorNode *NNGraph::TensorNode::add(
    NNGraph::TensorNode *rhs, Scalar alpha, Scalar beta) const
{
    return nntile::graph::add(
        alpha, const_cast<NNGraph::TensorNode *>(this), beta, rhs);
}

} // namespace nntile::graph
