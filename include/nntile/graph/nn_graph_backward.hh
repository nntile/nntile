/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph_backward.hh
 * Backward registry: maps OpType to backward builder.
 *
 * Each autograd op registers its build_backward; TensorNode::backward()
 * dispatches via this registry instead of centralizing logic in nn_graph.cc.
 *
 * @version 1.1.0
 * */

#pragma once

#include <functional>

#include <nntile/graph/logical_graph.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Signature for backward builder: add gradient ops for this op
using BackwardFn = std::function<void(
    NNGraph& graph,
    LogicalGraph::OpNode* op,
    NNGraph::TensorNode* grad_out)>;

//! Register backward for an OpType (called by op implementations)
void register_backward(OpType type, BackwardFn fn);

//! Get registered backward, or nullptr if none
BackwardFn get_backward(OpType type);

} // namespace nntile::graph
