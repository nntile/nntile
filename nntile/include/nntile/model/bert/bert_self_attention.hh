/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/bert/bert_self_attention.hh
 * BertSelfAttention - bidirectional multi-head self-attention (no output proj).
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/model/bert/bert_config.hh>
#include <nntile/module/module.hh>

namespace nntile::model::bert
{

//! Q/K/V projections and SDPA; output projection lives in BertSelfOutput.
class BertSelfAttention : public graph::module::Module
{
private:
    graph::NNGraph::TensorNode* w_q_ = nullptr;
    graph::NNGraph::TensorNode* w_k_ = nullptr;
    graph::NNGraph::TensorNode* w_v_ = nullptr;
    graph::NNGraph::TensorNode* q_bias_ = nullptr;
    graph::NNGraph::TensorNode* k_bias_ = nullptr;
    graph::NNGraph::TensorNode* v_bias_ = nullptr;

    BertConfig config_;
    graph::DataType dtype_;
    Index head_size_;
    Index n_heads_;

public:
    BertSelfAttention(graph::NNGraph* graph,
                      const std::string& name,
                      const BertConfig& config,
                      graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;
};

} // namespace nntile::model::bert
