/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gptneo/gptneo_causal.cc
 * GptneoCausal implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gptneo/gptneo_causal.hh"
#include "nntile/graph/nn/transpose.hh"

#include <stdexcept>

namespace nntile::model::gptneo
{

GptneoCausal::GptneoCausal(graph::NNGraph* graph,
                           const std::string& name,
                           const GptneoConfig& config,
                           graph::DataType dtype)
    : graph::module::Module(graph, name)
    , model_(std::make_unique<GptneoModel>(graph, name + "_model", config, dtype))
    , lm_head_(graph, name + "_lm_head",
               config.hidden_size, config.vocab_size, false, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("model", model_.get());
    register_module("lm_head", &lm_head_);
}

graph::NNGraph::TensorNode* GptneoCausal::forward(
    graph::NNGraph::TensorNode* input_ids,
    graph::NNGraph::TensorNode* position_ids,
    graph::NNGraph::TensorNode* mask)
{
    graph::NNGraph::TensorNode* hidden =
        model_->forward(input_ids, position_ids, mask);

    graph::NNGraph::TensorNode* hidden_t =
        graph::transpose(hidden, tensor_name("hidden_t"), 1);
    graph::NNGraph::TensorNode* logits_sbv = lm_head_.forward(hidden_t);
    return graph::transpose(logits_sbv, tensor_name("logits"), 2);
}

std::string GptneoCausal::repr() const
{
    return "GptneoCausal(" + model_->repr() + ", vocab=" +
           std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::gptneo
