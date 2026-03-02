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

#include "nntile/graph/nn_graph.hh"
#include "nntile/graph/grad_mode.hh"

namespace nntile::graph
{

void register_op(NNGraph& graph, std::shared_ptr<NNBaseOpNode> op)
{
    const bool need_backward =
        GradMode::is_enabled() && any_input_requires_grad(op->inputs());

    if(!need_backward)
    {
        return;
    }

    NNGraph::OpNode* op_nn = graph.create_op(std::move(op));

    for(NNGraph::TensorNode* out : op_nn->outputs())
    {
        if(out != nullptr)
        {
            out->set_producer(op_nn);
        }
    }
}

bool any_input_requires_grad(
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
