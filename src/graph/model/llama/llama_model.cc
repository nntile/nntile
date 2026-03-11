/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/model/llama/llama_model.cc
 * LlamaModel implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/llama/llama_model.hh"
#include "nntile/graph/nn/transpose.hh"

#include <stdexcept>

namespace nntile::model::llama
{

LlamaModel::LlamaModel(graph::NNGraph* graph,
                       const std::string& name,
                       const LlamaConfig& config,
                       graph::DataType dtype)
    : graph::module::Module(graph, name)
    , embed_tokens_(graph, name + "_embed_tokens",
                   config.vocab_size, config.hidden_size,
                   2, 0, dtype)  // axis=2 for (seq,batch) -> (seq,batch,hidden)
    , norm_(graph, name + "_norm",
            config.hidden_size, 0, config.rms_norm_eps, 0, dtype)  // axis=0 for (hidden,seq,batch)
    , config_(config)
    , dtype_(dtype)
{
    register_module("embed_tokens", &embed_tokens_);
    register_module("norm", &norm_);

    for(Index i = 0; i < config.num_hidden_layers; ++i)
    {
        auto layer = std::make_unique<LlamaDecoder>(
            graph, name + "_layers_" + std::to_string(i), config, dtype);
        register_module("layers_" + std::to_string(i), layer.get());
        layers_.push_back(std::move(layer));
    }
}

graph::NNGraph::TensorNode* LlamaModel::forward(
    graph::NNGraph::TensorNode* input_ids,
    graph::NNGraph::TensorNode* sin,
    graph::NNGraph::TensorNode* cos,
    graph::NNGraph::TensorNode* mask,
    const std::vector<std::pair<graph::NNGraph::TensorNode*,
                               graph::NNGraph::TensorNode*>>* kv_caches,
    Index cache_len)
{
    if(input_ids == nullptr)
    {
        throw std::invalid_argument(
            "LlamaModel::forward: input_ids must be non-null");
    }

    // Embedding: (seq, batch) -> (seq, batch, hidden)
    graph::NNGraph::TensorNode* embed = embed_tokens_.forward(input_ids);
    // Transpose to (hidden, seq, batch) for decoder layers (ndim=2)
    graph::NNGraph::TensorNode* x =
        graph::transpose(embed, tensor_name("embed_out"), 2);

    for(size_t i = 0; i < layers_.size(); ++i)
    {
        graph::NNGraph::TensorNode* k_cache = nullptr;
        graph::NNGraph::TensorNode* v_cache = nullptr;
        if(kv_caches != nullptr && i < kv_caches->size())
        {
            k_cache = (*kv_caches)[i].first;
            v_cache = (*kv_caches)[i].second;
        }
        x = layers_[i]->forward(x, sin, cos, mask, k_cache, v_cache, cache_len);
    }

    return norm_.forward(x);
}

std::string LlamaModel::repr() const
{
    return "LlamaModel(hidden=" + std::to_string(config_.hidden_size) +
           ", layers=" + std::to_string(config_.num_hidden_layers) + ")";
}

} // namespace nntile::model::llama
