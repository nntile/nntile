/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/autograd_function.hh
 * Base class for autograd functions - PyTorch-like.
 *
 * Handles OpNode creation, producer wiring, and requires_grad when GradMode
 * is enabled. Derived classes implement forward logic and build_backward.
 *
 * @version 1.1.0
 * */

#pragma once

#include <functional>
#include <vector>

#include <nntile/graph/grad_mode.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Base for autograd functors. Centralizes OpNode creation and producer wiring.
//!
//! Always creates OpNode. Sets producer and backward_fn only when GradMode
//! enabled and at least one output requires grad.
struct AutogradFunction
{
    //! Register OpNode (always created). Set producer and backward_fn only when
    //! GradMode enabled and any output requires grad.
    static void register_op(
        NNGraph& graph,
        const std::vector<NNGraph::TensorNode*>& inputs,
        const std::vector<NNGraph::TensorNode*>& outputs,
        OpAttrs attrs,
        std::function<void(const NNGraph::OpNode*)> backward_fn);

    //! Single-output overload
    static void register_op(
        NNGraph& graph,
        const std::vector<NNGraph::TensorNode*>& inputs,
        NNGraph::TensorNode* output,
        OpAttrs attrs,
        std::function<void(const NNGraph::OpNode*)> backward_fn);

    //! Compute output requires_grad: true if any input requires grad.
    static bool any_input_requires_grad(
        const std::vector<NNGraph::TensorNode*>& inputs);

    //! True if any output requires grad.
    static bool any_output_requires_grad(
        const std::vector<NNGraph::TensorNode*>& outputs);
};

} // namespace nntile::graph
