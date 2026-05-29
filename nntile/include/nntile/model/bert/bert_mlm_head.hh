/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/bert/bert_mlm_head.hh
 * BertMlmHead - prediction transform + vocab decoder (MLM).
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/model/bert/bert_config.hh>
#include <nntile/module/activation.hh>
#include <nntile/module/layer_norm.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/module.hh>

namespace nntile::model::bert
{

//! BertLMPredictionHead: transform (Linear+GELUTANH+LN) + decoder Linear.
class BertMlmHead : public module::Module
{
private:
    module::Linear transform_dense_;
    module::Activation transform_act_;
    module::LayerNorm transform_ln_;
    module::Linear decoder_;

    BertConfig config_;
    DataType dtype_;

public:
    BertMlmHead(NNGraph* graph,
                const std::string& name,
                const BertConfig& config,
                NNGraph::TensorNode* tied_word_vocab,
                DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(NNGraph::TensorNode* hidden);

    std::string repr() const override;
};

} // namespace nntile::model::bert
