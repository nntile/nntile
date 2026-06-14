/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/model/llama/llama_causal.cc
 * LlamaCausal implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/llama/llama_causal.hh"

#include <stdexcept>

namespace nntile::model::llama
{

LlamaCausal::LlamaCausal(NNGraph *graph,
    const std::string &name,
    const LlamaConfig &config,
    DataType dtype) :
    module::Module(graph, name),
    model_(
        std::make_unique<LlamaModel>(graph, name + "_model", config, dtype)),
    lm_head_(graph,
        name + "_lm_head",
        config.hidden_size,
        config.vocab_size,
        false,
        dtype),
    config_(config),
    dtype_(dtype)
{
    register_module("model", model_.get());
    register_module("lm_head", &lm_head_);
}

NNGraph::TensorNode *LlamaCausal::forward(
    NNGraph::TensorNode *input_ids,
    NNGraph::TensorNode *sin,
    NNGraph::TensorNode *cos,
    NNGraph::TensorNode *mask,
    KVCache *kv_cache)
{
    NNGraph::TensorNode *hidden =
        model_->forward(input_ids, sin, cos, mask, kv_cache);
    NNGraph::TensorNode *logits = lm_head_.forward(hidden);
    logits->set_name(tensor_name("logits"));
    return logits;
}

std::string LlamaCausal::repr() const
{
    return "LlamaCausal(" + model_->repr() +
           ", vocab=" + std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::llama
