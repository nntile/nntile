/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/t5/t5_for_conditional_generation.cc
 * T5ForConditionalGeneration implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/t5/t5_for_conditional_generation.hh"
#include "nntile/graph/nn/transpose.hh"

#include <stdexcept>

namespace nntile::model::t5
{

T5ForConditionalGeneration::T5ForConditionalGeneration(
    graph::NNGraph* graph,
    const std::string& name,
    const T5Config& config,
    graph::DataType dtype)
    : graph::module::Module(graph, name)
    , model_(std::make_unique<T5Model>(graph, name + "_model", config, dtype))
    , lm_head_(graph, name + "_lm_head",
               config.d_model, config.vocab_size, false, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("model", model_.get());
    register_module("lm_head", &lm_head_);
}

graph::NNGraph::TensorNode* T5ForConditionalGeneration::forward(
    graph::NNGraph::TensorNode* encoder_input_ids,
    graph::NNGraph::TensorNode* decoder_input_ids,
    graph::NNGraph::TensorNode* encoder_attention_mask,
    graph::NNGraph::TensorNode* decoder_attention_mask,
    graph::NNGraph::TensorNode* cross_attention_mask)
{
    // Model output: (d_model, dec_seq, batch)
    graph::NNGraph::TensorNode* hidden = model_->forward(
        encoder_input_ids, decoder_input_ids,
        encoder_attention_mask, decoder_attention_mask, cross_attention_mask);

    // Transpose (d_model, seq, batch) -> (seq, batch, d_model) for lm_head
    graph::NNGraph::TensorNode* hidden_t =
        graph::transpose(hidden, tensor_name("hidden_t"), 1);
    graph::NNGraph::TensorNode* logits_sbv = lm_head_.forward(hidden_t);
    // Transpose to (vocab, seq, batch) for output
    return graph::transpose(logits_sbv, tensor_name("logits"), 2);
}

std::string T5ForConditionalGeneration::repr() const
{
    return "T5ForConditionalGeneration(" + model_->repr() +
           ", vocab=" + std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::t5
