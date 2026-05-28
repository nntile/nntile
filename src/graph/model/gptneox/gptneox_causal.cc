/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gptneox/gptneox_causal.cc
 * GptneoxCausal implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gptneox/gptneox_causal.hh"
#include "nntile/graph/nn/ops/transpose.hh"

#include <stdexcept>

namespace nntile::graph::model::gptneox
{

GptneoxCausal::GptneoxCausal(graph::NNGraph* graph,
                             const std::string& name,
                             const GptneoxConfig& config,
                             graph::DataType dtype)
    : graph::module::Module(graph, name)
    , model_(std::make_unique<GptneoxModel>(graph, name + "_model", config, dtype))
    , lm_head_(graph, name + "_lm_head",
               config.hidden_size, config.vocab_size, false, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("model", model_.get());
    register_module("lm_head", &lm_head_);
}

graph::NNGraph::TensorNode* GptneoxCausal::forward(
    graph::NNGraph::TensorNode* input_ids,
    graph::NNGraph::TensorNode* sin,
    graph::NNGraph::TensorNode* cos,
    graph::NNGraph::TensorNode* mask)
{
    graph::NNGraph::TensorNode* hidden =
        model_->forward(input_ids, sin, cos, mask);
    graph::NNGraph::TensorNode* hidden_t = graph::transpose(hidden, 1);
    hidden_t->set_name(tensor_name("hidden_t"));
    graph::NNGraph::TensorNode* logits_sbv = lm_head_.forward(hidden_t);
    graph::NNGraph::TensorNode* logits = graph::transpose(logits_sbv, 2);
    logits->set_name(tensor_name("logits"));
    return logits;
}

std::string GptneoxCausal::repr() const
{
    return "GptneoxCausal(" + model_->repr() + ", vocab=" +
           std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::graph::model::gptneox
