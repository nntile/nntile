/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/concat.cc
 * NNGraph concat operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/concat.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>
#include <string>

#include "nntile/graph/tensor/concat.hh"

namespace nntile::graph
{

NNGraph::TensorNode* NNConcatOp::forward(const std::string& output_name)
{
    if(a == nullptr || b == nullptr)
    {
        throw std::invalid_argument(
            "NNConcatOp::forward: a and b must be non-null");
    }
    if(a->graph() != b->graph())
    {
        throw std::invalid_argument(
            "NNConcatOp::forward: a and b must belong to the same NNGraph");
    }
    NNGraph* graph = a->graph();
    const bool out_requires_grad = any_input_requires_grad({a, b});
    TensorGraph::TensorNode* out_data =
        graph::tensor::concat(a->data(), b->data(), axis, output_name);
    NNGraph::TensorNode* out = graph->tensor(out_data, out_requires_grad);
    outputs_ = {out};
    return out;
}

void NNConcatOp::backward() const
{
    NNGraph::TensorNode* out = output();
    if(out == nullptr || out->grad() == nullptr)
    {
        return;
    }
    throw std::runtime_error("NNGraph concat: backward is not supported yet");
}

NNGraph::TensorNode* concat(
    NNGraph::TensorNode* a,
    NNGraph::TensorNode* b,
    Index axis,
    const std::string& output_name)
{
    if(a == nullptr || b == nullptr)
    {
        throw std::invalid_argument("concat: input tensors must be non-null");
    }
    NNGraph* graph = a->graph();
    auto op = std::make_shared<NNConcatOp>(a, b, axis);
    NNGraph::TensorNode* out = op->forward(output_name);
    graph->register_op(std::move(op));
    return out;
}

} // namespace nntile::graph
