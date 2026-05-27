/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/bert/bert_attention.hh
 * BertAttention - self-attention + output projection with residual.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/graph/model/bert/bert_config.hh>
#include <nntile/graph/model/bert/bert_self_attention.hh>
#include <nntile/graph/model/bert/bert_self_output.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::graph::model::bert
{

class BertAttention : public graph::module::Module
{
private:
    BertSelfAttention self_attn_;
    BertSelfOutput self_out_;

    BertConfig config_;
    graph::DataType dtype_;

public:
    BertAttention(graph::NNGraph* graph,
                  const std::string& name,
                  const BertConfig& config,
                  graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;
};

} // namespace nntile::graph::model::bert
