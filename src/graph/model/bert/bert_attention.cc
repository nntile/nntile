/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/model/bert/bert_attention.cc
 * BertAttention implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/bert/bert_attention.hh"

namespace nntile::graph::model::bert
{

BertAttention::BertAttention(graph::NNGraph* graph,
                             const std::string& name,
                             const BertConfig& config,
                             graph::DataType dtype)
    : graph::module::Module(graph, name)
    , self_attn_(graph, name + "_self", config, dtype)
    , self_out_(graph, name + "_output", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("self", &self_attn_);
    register_module("output", &self_out_);
}

graph::NNGraph::TensorNode* BertAttention::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* mask,
    bool causal)
{
    graph::NNGraph::TensorNode* heads =
        self_attn_.forward(x, mask, causal);
    return self_out_.forward(heads, x);
}

std::string BertAttention::repr() const
{
    return "BertAttention(hidden=" + std::to_string(config_.hidden_size) + ")";
}

} // namespace nntile::graph::model::bert
