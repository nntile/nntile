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
#include <stdexcept>
#include <utility>
#include <vector>

#include <nntile/graph/grad_mode.hh>
#include <nntile/graph/logical_graph.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Result of build_forward: logical outputs + inputs + attrs for operator() bookkeeping.
struct ForwardResult
{
    std::vector<LogicalGraph::TensorNode*> outputs;
    std::vector<NNGraph::TensorNode*> inputs;
    OpAttrs attrs;
};

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

//! Base for autograd functors. operator() does all bookkeeping; build_forward does
//! only logical ops and returns ForwardResult.
template<typename Derived>
struct AutogradFunction : AutogradFunctionBase
{
    template<typename... Args>
    std::vector<NNGraph::TensorNode*> operator()(Args&&... args) const
    {
        ForwardResult result = Derived::build_forward(std::forward<Args>(args)...);
        if(result.inputs.empty())
        {
            throw std::invalid_argument(
                "AutogradFunction::operator(): build_forward must return non-empty inputs");
        }
        if(result.outputs.empty())
        {
            throw std::invalid_argument(
                "AutogradFunction::operator(): build_forward must return non-empty outputs");
        }
        NNGraph& graph = result.inputs[0]->graph();
        bool out_requires_grad = any_input_requires_grad(result.inputs);
        std::vector<NNGraph::TensorNode*> nn_outputs;
        nn_outputs.reserve(result.outputs.size());
        for(LogicalGraph::TensorNode* p : result.outputs)
        {
            if(p != nullptr)
            {
                nn_outputs.push_back(graph.tensor(*p, out_requires_grad));
            }
        }
        register_op(graph, result.inputs, nn_outputs, std::move(result.attrs),
                    [](const NNGraph::OpNode* op) {
                        Derived::build_backward(op);
                    });
        return nn_outputs;
    }
};

} // namespace nntile::graph
