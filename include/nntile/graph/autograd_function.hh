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
#include <utility>
#include <vector>

#include <nntile/graph/grad_mode.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Shared helpers for autograd functors (register_op, any_input_requires_grad).
struct AutogradFunctionBase
{
    static void register_op(
        NNGraph& graph,
        const std::vector<NNGraph::TensorNode*>& inputs,
        const std::vector<NNGraph::TensorNode*>& outputs,
        OpAttrs attrs,
        std::function<void(const NNGraph::OpNode*)> backward_fn);

    static void register_op(
        NNGraph& graph,
        const std::vector<NNGraph::TensorNode*>& inputs,
        NNGraph::TensorNode* output,
        OpAttrs attrs,
        std::function<void(const NNGraph::OpNode*)> backward_fn);

    static bool any_input_requires_grad(
        const std::vector<NNGraph::TensorNode*>& inputs);
};

//! Base for autograd functors. operator() forwards to Derived::build_forward.
//! User implements only build_forward() and build_backward().
template<typename Derived>
struct AutogradFunction : AutogradFunctionBase
{
    template<typename... Args>
    auto operator()(Args&&... args) const
    {
        return Derived::build_forward(std::forward<Args>(args)...);
    }
};

} // namespace nntile::graph
