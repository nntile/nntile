/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gpt2/gpt2_causal.cc
 * Gpt2Causal implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gpt2/gpt2_causal.hh"
#include "nntile/graph/nn/transpose.hh"

#include <stdexcept>

namespace nntile::model::gpt2
{

Gpt2Causal::Gpt2Causal(graph::NNGraph* graph,
                      const std::string& name,
                      const Gpt2Config& config,
                      graph::DataType dtype)
    : graph::module::Module(graph, name)
    , model_(std::make_unique<Gpt2Model>(graph, name + "_transformer", config, dtype))
    , lm_head_(graph, name + "_lm_head",
               config.hidden_size, config.vocab_size, false, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("transformer", model_.get());
    register_module("lm_head", &lm_head_);
}

graph::NNGraph::TensorNode* Gpt2Causal::forward(
    graph::NNGraph::TensorNode* input_ids,
    graph::NNGraph::TensorNode* position_ids,
    graph::NNGraph::TensorNode* mask)
{
    if(input_ids == nullptr)
    {
        throw std::invalid_argument(
            "Gpt2Causal::forward: input_ids must be non-null");
    }
    if(position_ids == nullptr)
    {
        throw std::invalid_argument(
            "Gpt2Causal::forward: position_ids must be non-null");
    }

    // Model output: (hidden, seq, batch)
    graph::NNGraph::TensorNode* hidden =
        model_->forward(input_ids, position_ids, mask);
    // Transpose (hidden, seq, batch) -> (seq, batch, hidden) for lm_head
    graph::NNGraph::TensorNode* hidden_t =
        graph::transpose(hidden, tensor_name("hidden_t"), 1);
    graph::NNGraph::TensorNode* logits_sbv = lm_head_.forward(hidden_t);
    // Transpose to (vocab, seq, batch) for output
    return graph::transpose(logits_sbv, tensor_name("logits"), 2);
}

std::string Gpt2Causal::repr() const
{
    return "Gpt2Causal(" + model_->repr() + ", vocab=" +
           std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::gpt2
