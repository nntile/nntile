/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/t5/t5_for_conditional_generation.cc
 * T5ForConditionalGeneration implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/t5/t5_for_conditional_generation.hh"
#include "nntile/nn_graph/ops/scale.hh"
#include "nntile/nn_graph/ops/transpose.hh"

#include <cmath>
#include <stdexcept>

namespace nntile::model::t5
{

T5ForConditionalGeneration::T5ForConditionalGeneration(
    NNGraph* graph,
    const std::string& name,
    const T5Config& config,
    DataType dtype)
    : module::Module(graph, name)
    , model_(std::make_unique<T5Model>(graph, name + "_model", config, dtype))
    , lm_head_(graph, name + "_lm_head",
               config.d_model, config.vocab_size, false, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("model", model_.get());
    register_module("lm_head", &lm_head_);
}

NNGraph::TensorNode* T5ForConditionalGeneration::forward(
    NNGraph::TensorNode* encoder_input_ids,
    NNGraph::TensorNode* decoder_input_ids,
    NNGraph::TensorNode* encoder_attention_mask,
    NNGraph::TensorNode* decoder_attention_mask,
    NNGraph::TensorNode* cross_attention_mask)
{
    // Model output: (d_model, dec_seq, batch)
    NNGraph::TensorNode* hidden = model_->forward(
        encoder_input_ids, decoder_input_ids,
        encoder_attention_mask, decoder_attention_mask, cross_attention_mask);

    // HF ``T5ForConditionalGeneration`` (``tie_word_embeddings=True``):
    // sequence_output *= d_model**-0.5 before lm_head projection.
    if(config_.tie_word_embeddings)
    {
        const Scalar inv_sqrt_d_model =
            1.f / std::sqrt(static_cast<Scalar>(config_.d_model));
        hidden = scale(inv_sqrt_d_model, hidden)
                     ->set_name(tensor_name("hidden_scaled"));
    }

    // Transpose (d_model, seq, batch) -> (seq, batch, d_model) for lm_head
    NNGraph::TensorNode* hidden_t = transpose(hidden, 1);
    hidden_t->set_name(tensor_name("hidden_t"));
    NNGraph::TensorNode* logits_sbv = lm_head_.forward(hidden_t);
    NNGraph::TensorNode* logits = transpose(logits_sbv, 2);
    logits->set_name(tensor_name("logits"));
    return logits;
}

std::string T5ForConditionalGeneration::repr() const
{
    return "T5ForConditionalGeneration(" + model_->repr() +
           ", vocab=" + std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::t5
