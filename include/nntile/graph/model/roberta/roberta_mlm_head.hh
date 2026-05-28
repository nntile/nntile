/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/roberta/roberta_mlm_head.hh
 * RobertaMlmHead - prediction transform + vocab decoder (MLM).
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/graph/model/roberta/roberta_config.hh>
#include <nntile/graph/module/activation.hh>
#include <nntile/graph/module/layer_norm.hh>
#include <nntile/graph/module/linear.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::graph::model::roberta
{

//! RobertaLMHead: dense + GELU + LN + decoder Linear + vocab bias.
class RobertaMlmHead : public graph::module::Module
{
private:
    graph::module::Linear transform_dense_;
    graph::module::Activation transform_act_;
    graph::module::LayerNorm transform_ln_;
    graph::module::Linear decoder_;
    graph::NNGraph::TensorNode* head_bias_tensor_ = nullptr;

    RobertaConfig config_;
    graph::DataType dtype_;

public:
    RobertaMlmHead(graph::NNGraph* graph,
                   const std::string& name,
                   const RobertaConfig& config,
                   graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(graph::NNGraph::TensorNode* hidden);

    std::string repr() const override;
};

} // namespace nntile::graph::model::roberta
