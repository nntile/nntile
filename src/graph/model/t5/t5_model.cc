/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/t5/t5_model.cc
 * T5Model implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/t5/t5_model.hh"
#include "nntile/graph/nn/transpose.hh"

#include <stdexcept>

namespace nntile::model::t5
{

T5Model::T5Model(graph::NNGraph* graph,
                const std::string& name,
                const T5Config& config,
                graph::DataType dtype)
    : graph::module::Module(graph, name)
    , embed_tokens_(graph, name + "_embed_tokens",
                    config.vocab_size, config.d_model,
                    2, 0, dtype)  // axis=2 for (seq,batch) -> (seq,batch,d_model)
    , encoder_final_norm_(graph, name + "_encoder_final_norm",
                          config.d_model, 0, config.layer_norm_epsilon, 0, dtype)
    , decoder_final_norm_(graph, name + "_decoder_final_norm",
                          config.d_model, 0, config.layer_norm_epsilon, 0, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("embed_tokens", &embed_tokens_);
    register_module("encoder_final_norm", &encoder_final_norm_);
    register_module("decoder_final_norm", &decoder_final_norm_);

    for(Index i = 0; i < config.num_layers; ++i)
    {
        auto layer = std::make_unique<T5EncoderBlock>(
            graph, name + "_encoder_layers_" + std::to_string(i), config, dtype);
        register_module("encoder_layers_" + std::to_string(i), layer.get());
        encoder_layers_.push_back(std::move(layer));
    }

    for(Index i = 0; i < config.num_decoder_layers; ++i)
    {
        auto layer = std::make_unique<T5DecoderBlock>(
            graph, name + "_decoder_layers_" + std::to_string(i), config, dtype);
        register_module("decoder_layers_" + std::to_string(i), layer.get());
        decoder_layers_.push_back(std::move(layer));
    }
}

graph::NNGraph::TensorNode* T5Model::forward(
    graph::NNGraph::TensorNode* encoder_input_ids,
    graph::NNGraph::TensorNode* decoder_input_ids,
    graph::NNGraph::TensorNode* encoder_attention_mask,
    graph::NNGraph::TensorNode* decoder_attention_mask,
    graph::NNGraph::TensorNode* cross_attention_mask)
{
    if(encoder_input_ids == nullptr || decoder_input_ids == nullptr)
    {
        throw std::invalid_argument(
            "T5Model::forward: encoder_input_ids and decoder_input_ids must be non-null");
    }

    // Shared embedding
    graph::NNGraph::TensorNode* encoder_embed =
        embed_tokens_.forward(encoder_input_ids);
    graph::NNGraph::TensorNode* decoder_embed =
        embed_tokens_.forward(decoder_input_ids);

    // Transpose to (d_model, seq, batch)
    graph::NNGraph::TensorNode* encoder_x =
        graph::transpose(encoder_embed, tensor_name("encoder_x"), 2);
    graph::NNGraph::TensorNode* decoder_x =
        graph::transpose(decoder_embed, tensor_name("decoder_x"), 2);

    // Encoder stack
    graph::NNGraph::TensorNode* enc_hidden = encoder_x;
    for(auto& layer : encoder_layers_)
    {
        enc_hidden = layer->forward(enc_hidden, encoder_attention_mask);
    }
    graph::NNGraph::TensorNode* encoder_hidden_states =
        encoder_final_norm_.forward(enc_hidden);

    // Decoder stack
    graph::NNGraph::TensorNode* dec_hidden = decoder_x;
    for(auto& layer : decoder_layers_)
    {
        dec_hidden = layer->forward(
            dec_hidden, encoder_hidden_states,
            decoder_attention_mask, cross_attention_mask);
    }
    return decoder_final_norm_.forward(dec_hidden);
}

std::string T5Model::repr() const
{
    return "T5Model(d_model=" + std::to_string(config_.d_model) +
           ", encoder_layers=" + std::to_string(config_.num_layers) +
           ", decoder_layers=" + std::to_string(config_.num_decoder_layers) + ")";
}

} // namespace nntile::model::t5
