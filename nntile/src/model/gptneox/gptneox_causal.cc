/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/gptneox/gptneox_causal.cc
 * GptneoxCausal implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gptneox/gptneox_causal.hh"

#include <stdexcept>

namespace nntile::model::gptneox
{

GptneoxCausal::GptneoxCausal(NNGraph* graph,
                             const std::string& name,
                             const GptneoxConfig& config,
                             DataType dtype)
    : module::Module(graph, name)
    , model_(std::make_unique<GptneoxModel>(graph, name + "_model", config, dtype))
    , lm_head_(graph, name + "_lm_head",
               config.hidden_size, config.vocab_size, false, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("model", model_.get());
    register_module("lm_head", &lm_head_);
}

NNGraph::TensorNode* GptneoxCausal::forward(
    NNGraph::TensorNode* input_ids,
    NNGraph::TensorNode* sin,
    NNGraph::TensorNode* cos,
    NNGraph::TensorNode* mask)
{
    NNGraph::TensorNode* hidden =
        model_->forward(input_ids, sin, cos, mask);
    NNGraph::TensorNode* logits = lm_head_.forward(hidden);
    logits->set_name(tensor_name("logits"));
    return logits;
}

std::string GptneoxCausal::repr() const
{
    return "GptneoxCausal(" + model_->repr() + ", vocab=" +
           std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::gptneox
