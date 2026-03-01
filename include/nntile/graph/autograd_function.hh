/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/autograd_function.hh
 * Helpers for autograd: register_op, any_input_requires_grad.
 *
 * No CRTP. User performs all bookkeeping in build_forward.
 *
 * @version 1.1.0
 * */

#pragma once

#include <functional>
#include <vector>

#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Register OpNode. Creates only when GradMode enabled and any input requires grad.
void register_op(
    NNGraph& graph,
    const std::vector<NNGraph::TensorNode*>& inputs,
    const std::vector<NNGraph::TensorNode*>& outputs,
    OpAttrs attrs,
    std::function<void(const NNGraph::OpNode*)> backward_fn,
    const std::vector<NNGraph::TensorNode*>& buffers = {});

void register_op(
    NNGraph& graph,
    const std::vector<NNGraph::TensorNode*>& inputs,
    NNGraph::TensorNode* output,
    OpAttrs attrs,
    std::function<void(const NNGraph::OpNode*)> backward_fn,
    const std::vector<NNGraph::TensorNode*>& buffers = {});

//! True if any input requires grad.
bool any_input_requires_grad(
    const std::vector<NNGraph::TensorNode*>& inputs);

} // namespace nntile::graph
