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
    NNGraph::TensorNode* output,
    OpAttrs attrs,
    std::function<void(const NNGraph::OpNode*)> backward_fn)
{
    if(!GradMode::is_enabled())
    {
        return;
    }
    NNGraph::OpNode* op_nn = graph.create_op(
        std::vector<NNGraph::TensorNode*>(inputs),
        {output},
        std::move(attrs),
        std::move(backward_fn));
    output->set_producer(op_nn);
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

} // namespace nntile::graph
