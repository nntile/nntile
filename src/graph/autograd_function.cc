/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/autograd_function.cc
 * AutogradFunction implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/autograd_function.hh"

namespace nntile::graph
{

void AutogradFunction::register_op(
    NNGraph& graph,
    const std::vector<NNGraph::TensorNode*>& inputs,
    const std::vector<NNGraph::TensorNode*>& outputs,
    OpAttrs attrs,
    std::function<void(const NNGraph::OpNode*)> backward_fn)
{
    const bool need_backward =
        GradMode::is_enabled() && any_output_requires_grad(outputs);

    NNGraph::OpNode* op_nn = graph.create_op(
        std::vector<NNGraph::TensorNode*>(inputs),
        std::vector<NNGraph::TensorNode*>(outputs),
        std::move(attrs),
        need_backward ? std::move(backward_fn)
                      : std::function<void(const NNGraph::OpNode*)>{});

    if(need_backward)
    {
        for(NNGraph::TensorNode* out : outputs)
        {
            if(out != nullptr)
            {
                out->set_producer(op_nn);
            }
        }
    }
}

void AutogradFunction::register_op(
    NNGraph& graph,
    const std::vector<NNGraph::TensorNode*>& inputs,
    NNGraph::TensorNode* output,
    OpAttrs attrs,
    std::function<void(const NNGraph::OpNode*)> backward_fn)
{
    register_op(graph, inputs, output ? std::vector<NNGraph::TensorNode*>{output}
                                      : std::vector<NNGraph::TensorNode*>{},
                std::move(attrs), std::move(backward_fn));
}

bool AutogradFunction::any_input_requires_grad(
    const std::vector<NNGraph::TensorNode*>& inputs)
{
    for(const auto* in : inputs)
    {
        if(in != nullptr && in->requires_grad())
        {
            return true;
        }
    }
    return false;
}

bool AutogradFunction::any_output_requires_grad(
    const std::vector<NNGraph::TensorNode*>& outputs)
{
    for(const auto* out : outputs)
    {
        if(out != nullptr && out->requires_grad())
        {
            return true;
        }
    }
    return false;
}

} // namespace nntile::graph
