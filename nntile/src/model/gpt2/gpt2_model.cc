#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/gpt2/gpt2_model.cc
 * GPT2Model implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gpt2/gpt2_model.hh"
#include "nntile/nn/ops/add.hh"

#include <stdexcept>

namespace nntile::model::gpt2
{

Gpt2Model::Gpt2Model(NNGraph* graph,
                    const std::string& name,
                    const Gpt2Config& config,
                    DataType dtype)
    : module::Module(graph, name)
    , wte_(graph, name + "_wte",
           config.vocab_size, config.hidden_size, dtype)
    , wpe_(graph, name + "_wpe",
           config.max_position_embeddings, config.hidden_size, dtype)
    , ln_f_(graph, name + "_ln_f",
            config.hidden_size, 2, config.layer_norm_eps, 0, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
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

NNGraph::TensorNode* Gpt2Model::forward(
    NNGraph::TensorNode* input_ids,
    NNGraph::TensorNode* position_ids,
    NNGraph::TensorNode* mask,
    bool causal)
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

    NNGraph::TensorNode* wte_out = wte_.forward(input_ids);
    NNGraph::TensorNode* wpe_out = wpe_.forward(position_ids);
    // Embeddings: (batch, seq) -> (batch, seq, hidden); sum token+position
    NNGraph::TensorNode* x =
        add(1.0, wte_out, 1.0, wpe_out);

    for(auto& layer : layers_)
    {
        x = layer->forward(x, mask, causal);
    }

    return ln_f_.forward(x);
}

std::string Gpt2Model::repr() const
{
    return "Gpt2Model(hidden=" + std::to_string(config_.hidden_size) +
           ", layers=" + std::to_string(config_.num_hidden_layers) + ")";
}

} // namespace nntile::model::gpt2
