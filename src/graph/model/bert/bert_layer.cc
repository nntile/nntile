/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/model/bert/bert_layer.cc
 * BertLayer implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/bert/bert_layer.hh"

#include <stdexcept>

namespace nntile::graph::model::bert
{

BertLayer::BertLayer(graph::NNGraph* graph,
                     const std::string& name,
                     const BertConfig& config,
                     graph::DataType dtype)
    : graph::module::Module(graph, name)
    , attention_(graph, name + "_attention", config, dtype)
    , intermediate_(graph, name + "_intermediate", config, dtype)
    , output_(graph, name + "_output", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("attention", &attention_);
    register_module("intermediate", &intermediate_);
    register_module("output", &output_);
}

graph::NNGraph::TensorNode* BertLayer::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* mask,
    bool causal)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "BertLayer::forward: input tensor must be non-null");
    }

    graph::NNGraph::TensorNode* attn_out =
        attention_.forward(x, mask, causal);
    graph::NNGraph::TensorNode* inter =
        intermediate_.forward(attn_out);
    return output_.forward(inter, attn_out);
}

std::string BertLayer::repr() const
{
    return "BertLayer(hidden=" + std::to_string(config_.hidden_size) + ")";
}

} // namespace nntile::graph::model::bert
