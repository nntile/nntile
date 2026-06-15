/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/bert/bert_self_output.hh
 * BertSelfOutput - dense projection, residual, layer norm.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/model/bert/bert_config.hh>
#include <nntile/module/layer_norm.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/module.hh>

namespace nntile::model::bert
{

//! Maps attention heads to hidden, adds residual, applies LayerNorm.
class BertSelfOutput : public module::Module
{
private:
    NNGraph::TensorNode* w_dense_ = nullptr;
    NNGraph::TensorNode* b_dense_ = nullptr;
    module::LayerNorm layer_norm_;

    BertConfig config_;
    DataType dtype_;

public:
    BertSelfOutput(NNGraph* graph,
                   const std::string& name,
                   const BertConfig& config,
                   DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* attn_heads,
        NNGraph::TensorNode* residual);

    std::string repr() const override;
};

} // namespace nntile::model::bert
