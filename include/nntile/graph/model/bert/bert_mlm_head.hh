/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/bert/bert_mlm_head.hh
 * BertMlmHead - prediction transform + vocab decoder (MLM).
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/graph/model/bert/bert_config.hh>
#include <nntile/graph/module/activation.hh>
#include <nntile/graph/module/layer_norm.hh>
#include <nntile/graph/module/linear.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::graph::model::bert
{

//! BertLMPredictionHead: transform (Linear+GELUTANH+LN) + decoder Linear.
class BertMlmHead : public graph::module::Module
{
private:
    graph::module::Linear transform_dense_;
    graph::module::Activation transform_act_;
    graph::module::LayerNorm transform_ln_;
    graph::module::Linear decoder_;

    BertConfig config_;
    graph::DataType dtype_;

public:
    BertMlmHead(graph::NNGraph* graph,
                const std::string& name,
                const BertConfig& config,
                graph::NNGraph::TensorNode* tied_word_vocab,
                graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(graph::NNGraph::TensorNode* hidden);

    std::string repr() const override;
};

} // namespace nntile::graph::model::bert
