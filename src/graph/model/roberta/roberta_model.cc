/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/model/roberta/roberta_model.cc
 * RobertaModel implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/roberta/roberta_model.hh"
#include "nntile/graph/model/roberta/roberta_common.hh"

#include <stdexcept>

namespace nntile::model::roberta
{

RobertaModel::RobertaModel(graph::NNGraph* graph,
                           const std::string& name,
                           const RobertaConfig& config,
                           graph::DataType dtype)
    : graph::module::Module(graph, name)
    , embeddings_(graph, name + "_embeddings", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("embeddings", &embeddings_);

    const bert::BertConfig bert_cfg = to_bert_config(config_);
    for(Index i = 0; i < config_.num_hidden_layers; ++i)
    {
        auto layer = std::make_unique<bert::BertLayer>(
            graph, name + "_layer_" + std::to_string(i), bert_cfg, dtype);
        register_module("layer_" + std::to_string(i), layer.get());
        layers_.push_back(std::move(layer));
    }
}

graph::NNGraph::TensorNode* RobertaModel::forward(
    graph::NNGraph::TensorNode* input_ids,
    graph::NNGraph::TensorNode* position_ids,
    graph::NNGraph::TensorNode* mask,
    bool causal)
{
    if(input_ids == nullptr || position_ids == nullptr)
    {
        throw std::invalid_argument(
            "RobertaModel::forward: input_ids and position_ids must be "
            "non-null");
    }
    throw_if_causal_flag_set(causal, "RobertaModel");

    graph::NNGraph::TensorNode* x =
        embeddings_.forward(input_ids, position_ids);

    for(auto& layer : layers_)
    {
        x = layer->forward(x, mask, causal);
    }
    return x;
}

std::string RobertaModel::repr() const
{
    return "RobertaModel(hidden=" + std::to_string(config_.hidden_size) +
           ", layers=" + std::to_string(config_.num_hidden_layers) + ")";
}

} // namespace nntile::model::roberta
