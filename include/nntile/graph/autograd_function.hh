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
//! Usage: Derived::build_forward does forward, then calls register_op().
//! When GradMode is enabled, creates OpNode and sets producer on output.
struct AutogradFunction
{
    //! Register OpNode and set producer on output when GradMode enabled.
    //! Call after creating output tensor in build_forward.
    static void register_op(
        NNGraph& graph,
        const std::vector<NNGraph::TensorNode*>& inputs,
        NNGraph::TensorNode* output,
        OpAttrs attrs,
        std::function<void(const NNGraph::OpNode*)> backward_fn);

    //! Compute output requires_grad: true if any input requires grad.
    static bool any_input_requires_grad(
        const std::vector<NNGraph::TensorNode*>& inputs);
};

} // namespace nntile::graph
