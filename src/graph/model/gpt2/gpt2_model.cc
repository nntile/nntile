/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gpt2/gpt2_model.cc
 * GPT2Model implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gpt2/gpt2_model.hh"
#include "nntile/graph/nn/add.hh"
#include "nntile/graph/nn/transpose.hh"

#include <stdexcept>

namespace nntile::model::gpt2
{

Gpt2Model::Gpt2Model(graph::NNGraph* graph,
                    const std::string& name,
                    const Gpt2Config& config,
                    graph::DataType dtype)
    : graph::module::Module(graph, name)
    , wte_(graph, name + "_wte",
           config.vocab_size, config.hidden_size,
           2, 0, dtype)
    , wpe_(graph, name + "_wpe",
           config.max_position_embeddings, config.hidden_size,
           2, 0, dtype)
    , ln_f_(graph, name + "_ln_f",
            config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("wte", &wte_);
    register_module("wpe", &wpe_);
    register_module("ln_f", &ln_f_);

    for(Index i = 0; i < config.num_hidden_layers; ++i)
    {
        auto layer = std::make_unique<Gpt2Block>(
            graph, name + "_h_" + std::to_string(i), config, dtype);
        register_module("h_" + std::to_string(i), layer.get());
        layers_.push_back(std::move(layer));
    }
}

graph::NNGraph::TensorNode* Gpt2Model::forward(
    graph::NNGraph::TensorNode* input_ids,
    graph::NNGraph::TensorNode* position_ids,
    graph::NNGraph::TensorNode* mask)
{
    if(input_ids == nullptr)
    {
        throw std::invalid_argument(
            "Gpt2Model::forward: input_ids must be non-null");
    }
    if(position_ids == nullptr)
    {
        throw std::invalid_argument(
            "Gpt2Model::forward: position_ids must be non-null");
    }

    // wte: (seq, batch) -> (seq, batch, hidden)
    graph::NNGraph::TensorNode* wte_out = wte_.forward(input_ids);
    // wpe: (seq, batch) -> (seq, batch, hidden)
    graph::NNGraph::TensorNode* wpe_out = wpe_.forward(position_ids);
    // add: wte + wpe
    graph::NNGraph::TensorNode* embed =
        graph::add(1.0, wte_out, 1.0, wpe_out, tensor_name("embed"));
    // Transpose to (hidden, seq, batch) for decoder layers
    graph::NNGraph::TensorNode* x =
        graph::transpose(embed, tensor_name("embed_t"), 2);

    for(auto& layer : layers_)
    {
        x = layer->forward(x, mask);
    }

    return ln_f_.forward(x);
}

std::string Gpt2Model::repr() const
{
    return "Gpt2Model(hidden=" + std::to_string(config_.hidden_size) +
           ", layers=" + std::to_string(config_.num_hidden_layers) + ")";
}

} // namespace nntile::model::gpt2
