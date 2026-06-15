/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/bert/bert_intermediate.cc
 * BertIntermediate implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/bert/bert_intermediate.hh"
#include "nntile/model/bert/bert_config.hh"

namespace nntile::model::bert
{

BertIntermediate::BertIntermediate(NNGraph* graph,
                                   const std::string& name,
                                   const BertConfig& config,
                                   DataType dtype)
    : module::Module(graph, name)
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

NNGraph::TensorNode* BertIntermediate::forward(
    NNGraph::TensorNode* x)
{
    NNGraph::TensorNode* hidden = dense_.forward(x);
    return activation_.forward(hidden);
}

std::string BertIntermediate::repr() const
{
    return "BertIntermediate(hidden=" +
           std::to_string(config_.hidden_size) +
           ", intermediate=" + std::to_string(config_.intermediate_size) + ")";
}

} // namespace nntile::model::bert
