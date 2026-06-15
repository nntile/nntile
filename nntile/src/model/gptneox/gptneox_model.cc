#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/gptneox/gptneox_model.cc
 * GptneoxModel implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gptneox/gptneox_model.hh"

#include <stdexcept>

namespace nntile::model::gptneox
{

GptneoxModel::GptneoxModel(NNGraph* graph,
                           const std::string& name,
                           const GptneoxConfig& config,
                           DataType dtype)
    : module::Module(graph, name)
    , embed_tokens_(graph, name + "_embed_tokens",
                    config.vocab_size, config.hidden_size, dtype)
    , norm_(graph, name + "_norm",
            config.hidden_size, 2, config.layer_norm_eps, 0, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("embed_tokens", &embed_tokens_);
    register_module("norm", &norm_);

    for(Index i = 0; i < config.num_hidden_layers; ++i)
    {
        auto layer = std::make_unique<GptneoxDecoder>(
            graph, name + "_layers_" + std::to_string(i), config, dtype);
        register_module("layers_" + std::to_string(i), layer.get());
        layers_.push_back(std::move(layer));
    }
}

NNGraph::TensorNode* GptneoxModel::forward(
    NNGraph::TensorNode* input_ids,
    NNGraph::TensorNode* sin,
    NNGraph::TensorNode* cos,
    NNGraph::TensorNode* mask)
{
    if(input_ids == nullptr)
    {
        throw std::invalid_argument(
            "GptneoxModel::forward: input_ids must be non-null");
    }

    NNGraph::TensorNode* x = embed_tokens_.forward(input_ids);
    x->set_name(tensor_name("embed_out"));

    for(auto& layer : layers_)
    {
        x = layer->forward(x, sin, cos, mask);
    }

    return norm_.forward(x);
}

std::string GptneoxModel::repr() const
{
    return "GptneoxModel(hidden=" + std::to_string(config_.hidden_size) +
           ", layers=" + std::to_string(config_.num_hidden_layers) + ")";
}

} // namespace nntile::model::gptneox
