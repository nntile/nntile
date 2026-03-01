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
#include <nntile/graph/logical_graph.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Base for autograd functors. Centralizes OpNode creation and producer wiring.
//!
//! Always creates OpNode. Sets producer and backward_fn only when GradMode
//! enabled and any input requires grad.
struct AutogradFunction
{
    //! Convenience wrapper (single output): run forward_fn, wrap output, register_op.
    //!
    //! @param forward_fn Callable () -> LogicalGraph::TensorNode& (the logical op)
    //! @param backward_fn Callable (OpNode*) -> void
    //! @return NNGraph::TensorNode* wrapping the logical output
    template<typename FwdFn, typename BwdFn>
    static NNGraph::TensorNode* run(
        NNGraph& graph,
        const std::vector<NNGraph::TensorNode*>& inputs,
        OpAttrs attrs,
        FwdFn&& forward_fn,
        BwdFn&& backward_fn)
    {
        LogicalGraph::TensorNode& out_data = forward_fn();
        bool out_requires_grad = any_input_requires_grad(inputs);
        NNGraph::TensorNode* output = graph.tensor(out_data, out_requires_grad);
        register_op(graph, inputs, output, std::move(attrs),
                    std::forward<BwdFn>(backward_fn));
        return output;
    }

    //! Convenience wrapper (multi-output): run forward_fn, wrap outputs, register_op.
    //!
    //! @param forward_fn Callable () -> std::vector<LogicalGraph::TensorNode*>
    //!                   (pointers to logical outputs, e.g. {&out1, &out2})
    //! @param backward_fn Callable (OpNode*) -> void
    //! @return std::vector<NNGraph::TensorNode*> wrapping the logical outputs
    template<typename FwdFn, typename BwdFn>
    static std::vector<NNGraph::TensorNode*> run_multi(
        NNGraph& graph,
        const std::vector<NNGraph::TensorNode*>& inputs,
        OpAttrs attrs,
        FwdFn&& forward_fn,
        BwdFn&& backward_fn)
    {
        std::vector<LogicalGraph::TensorNode*> out_data_vec = forward_fn();
        bool out_requires_grad = any_input_requires_grad(inputs);
        std::vector<NNGraph::TensorNode*> outputs;
        outputs.reserve(out_data_vec.size());
        for(LogicalGraph::TensorNode* p : out_data_vec)
        {
            if(p != nullptr)
            {
                outputs.push_back(graph.tensor(*p, out_requires_grad));
            }
        }
        register_op(graph, inputs, outputs, std::move(attrs),
                    std::forward<BwdFn>(backward_fn));
        return outputs;
    }

    //! Register OpNode (always created). Set producer and backward_fn only when
    //! GradMode enabled and any input requires grad.
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

};

} // namespace nntile::graph
