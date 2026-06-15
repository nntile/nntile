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
class BertSelfAttention : public module::Module
{
private:
    NNGraph::TensorNode* w_q_ = nullptr;
    NNGraph::TensorNode* w_k_ = nullptr;
    NNGraph::TensorNode* w_v_ = nullptr;
    NNGraph::TensorNode* q_bias_ = nullptr;
    NNGraph::TensorNode* k_bias_ = nullptr;
    NNGraph::TensorNode* v_bias_ = nullptr;

    BertConfig config_;
    DataType dtype_;
    Index head_size_;
    Index n_heads_;

public:
    BertSelfAttention(NNGraph* graph,
                      const std::string& name,
                      const BertConfig& config,
                      DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* x,
        NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;
};

} // namespace nntile::model::bert
