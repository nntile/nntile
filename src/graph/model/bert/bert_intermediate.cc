/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/model/bert/bert_intermediate.cc
 * BertIntermediate implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/bert/bert_intermediate.hh"

namespace nntile::model::bert
{

BertIntermediate::BertIntermediate(graph::NNGraph* graph,
                                   const std::string& name,
                                   const BertConfig& config,
                                   graph::DataType dtype)
    : graph::module::Module(graph, name)
    , dense_(graph, name + "_dense",
             config.hidden_size,
             config.intermediate_size,
             true,
             dtype)
    , activation_(graph, name + "_act",
                  graph::module::ActivationType::GELUTANH,
                  dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("dense", &dense_);
    register_module("act", &activation_);
}

graph::NNGraph::TensorNode* BertIntermediate::forward(
    graph::NNGraph::TensorNode* x)
{
    graph::NNGraph::TensorNode* hidden = dense_.forward(x);
    return activation_.forward(hidden);
}

std::string BertIntermediate::repr() const
{
    return "BertIntermediate(hidden=" +
           std::to_string(config_.hidden_size) +
           ", intermediate=" + std::to_string(config_.intermediate_size) + ")";
}

} // namespace nntile::model::bert
