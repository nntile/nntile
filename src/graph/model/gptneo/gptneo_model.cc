/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gptneo/gptneo_model.cc
 * GPT-Neo model implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gptneo/gptneo_model.hh"
#include "nntile/graph/nn/add.hh"
#include "nntile/graph/nn/transpose.hh"

#include <stdexcept>

namespace nntile::model::gptneo
{

GptneoModel::GptneoModel(graph::NNGraph* graph,
                         const std::string& name,
                         const GptneoConfig& config,
                         graph::DataType dtype)
    : graph::module::Module(graph, name)
    , wte_(graph, name + "_wte",
           config.vocab_size, config.hidden_size,
           2, 0, dtype)
    , wpe_(graph, name + "_wpe",
           config.max_position_embeddings, config.hidden_size,
           2, 0, dtype)
    , norm_(graph, name + "_norm",
            config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("wte", &wte_);
    register_module("wpe", &wpe_);
    register_module("norm", &norm_);

    for(Index i = 0; i < config.num_hidden_layers; ++i)
    {
        auto layer = std::make_unique<GptneoDecoder>(
            graph, name + "_layers_" + std::to_string(i), config, dtype);
        register_module("layers_" + std::to_string(i), layer.get());
        layers_.push_back(std::move(layer));
    }
}

graph::NNGraph::TensorNode* GptneoModel::forward(
    graph::NNGraph::TensorNode* input_ids,
    graph::NNGraph::TensorNode* position_ids,
    graph::NNGraph::TensorNode* mask)
{
    if(input_ids == nullptr)
    {
        throw std::invalid_argument(
            "GptneoModel::forward: input_ids must be non-null");
    }

    graph::NNGraph::TensorNode* token_embed = wte_.forward(input_ids);

    graph::NNGraph::TensorNode* pos_embed;
    if(position_ids != nullptr)
    {
        pos_embed = wpe_.forward(position_ids);
    }
    else
    {
        throw std::invalid_argument(
            "GptneoModel::forward: position_ids required for GPT-Neo");
    }

    graph::NNGraph::TensorNode* embed =
        graph::add(1.0, token_embed, 1.0, pos_embed, tensor_name("embed"));

    graph::NNGraph::TensorNode* x =
        graph::transpose(embed, tensor_name("embed_out"), 2);

    for(auto& layer : layers_)
    {
        x = layer->forward(x, mask);
    }

    return norm_.forward(x);
}

std::string GptneoModel::repr() const
{
    return "GptneoModel(hidden=" + std::to_string(config_.hidden_size) +
           ", layers=" + std::to_string(config_.num_hidden_layers) + ")";
}

} // namespace nntile::model::gptneo
