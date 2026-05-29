/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/model/bert/bert_intermediate.cc
 * BertIntermediate implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/bert/bert_intermediate.hh"
#include "nntile/model/bert/bert_config.hh"
#include "nntile/nn_graph/ops/add_fiber.hh"
#include "nntile/nn_graph/ops/gemm.hh"

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
                  activation_type_from_config(config))
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
    graph::NNGraph::TensorNode* hidden = graph::gemm(
        dense_.weight_tensor(),
        x,
        1.0,
        true,
        false,
        1,
        0);
    hidden = graph::add_fiber(1.0, dense_.bias_tensor(), 1.0, hidden, 0, 0);
    return activation_.forward(hidden);
}

std::string BertIntermediate::repr() const
{
    return "BertIntermediate(hidden=" +
           std::to_string(config_.hidden_size) +
           ", intermediate=" + std::to_string(config_.intermediate_size) + ")";
}

} // namespace nntile::model::bert
