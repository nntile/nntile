#include <nntile/graph/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/bert/bert_model.cc
 * BertModel implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/bert/bert_model.hh"

#include <stdexcept>

namespace nntile::graph::model::bert
{

BertModel::BertModel(graph::NNGraph* graph,
                     const std::string& name,
                     const BertConfig& config,
                     graph::DataType dtype)
    : graph::module::Module(graph, name)
    , embeddings_(graph, name + "_embeddings", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("embeddings", &embeddings_);

    for(Index i = 0; i < config.num_hidden_layers; ++i)
    {
        auto layer = std::make_unique<BertLayer>(
            graph, name + "_layer_" + std::to_string(i), config, dtype);
        register_module("layer_" + std::to_string(i), layer.get());
        layers_.push_back(std::move(layer));
    }
}

graph::NNGraph::TensorNode* BertModel::forward(
    graph::NNGraph::TensorNode* input_ids,
    graph::NNGraph::TensorNode* token_type_ids,
    graph::NNGraph::TensorNode* position_ids,
    graph::NNGraph::TensorNode* mask,
    bool causal)
{
    if(input_ids == nullptr || token_type_ids == nullptr ||
       position_ids == nullptr)
    {
        throw std::invalid_argument(
            "BertModel::forward: input_ids, token_type_ids, and "
            "position_ids must be non-null");
    }

    graph::NNGraph::TensorNode* x = embeddings_.forward(
        input_ids, token_type_ids, position_ids);

    for(auto& layer : layers_)
    {
        x = layer->forward(x, mask, causal);
    }
    return x;
}

std::string BertModel::repr() const
{
    return "BertModel(hidden=" + std::to_string(config_.hidden_size) +
           ", layers=" + std::to_string(config_.num_hidden_layers) + ")";
}

} // namespace nntile::graph::model::bert
