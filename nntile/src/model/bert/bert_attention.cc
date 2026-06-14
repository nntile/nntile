/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/model/bert/bert_attention.cc
 * BertAttention implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/bert/bert_attention.hh"

namespace nntile::model::bert
{

BertAttention::BertAttention(NNGraph* graph,
                             const std::string& name,
                             const BertConfig& config,
                             DataType dtype)
    : module::Module(graph, name)
    , self_attn_(graph, name + "_self", config, dtype)
    , self_out_(graph, name + "_output", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("self", &self_attn_);
    register_module("output", &self_out_);
}

NNGraph::TensorNode* BertAttention::forward(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* mask,
    bool causal)
{
    NNGraph::TensorNode* dense =
        self_attn_.forward(x, mask, causal,
            self_out_.w_dense(), self_out_.b_dense());
    return self_out_.forward(dense, x);
}

std::string BertAttention::repr() const
{
    return "BertAttention(hidden=" + std::to_string(config_.hidden_size) + ")";
}

} // namespace nntile::model::bert
