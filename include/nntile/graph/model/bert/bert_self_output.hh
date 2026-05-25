/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/bert/bert_self_output.hh
 * BertSelfOutput - dense projection, residual, layer norm.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/graph/model/bert/bert_config.hh>
#include <nntile/graph/module/layer_norm.hh>
#include <nntile/graph/module/linear.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::model::bert
{

//! Maps attention heads to hidden, adds residual, applies LayerNorm.
class BertSelfOutput : public graph::module::Module
{
private:
    graph::NNGraph::TensorNode* w_dense_ = nullptr;
    graph::NNGraph::TensorNode* b_dense_ = nullptr;
    graph::module::LayerNorm layer_norm_;

    BertConfig config_;
    graph::DataType dtype_;

public:
    BertSelfOutput(graph::NNGraph* graph,
                   const std::string& name,
                   const BertConfig& config,
                   graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* attn_heads,
        graph::NNGraph::TensorNode* residual);

    std::string repr() const override;
};

} // namespace nntile::model::bert
