/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/roberta/roberta_mlm_head.cc
 * RobertaMlmHead implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/roberta/roberta_mlm_head.hh"
#include "nntile/nn/ops/add_fiber.hh"

namespace nntile::model::roberta
{

RobertaMlmHead::RobertaMlmHead(NNGraph* graph,
                               const std::string& name,
                               const RobertaConfig& config,
                               DataType dtype)
    : module::Module(graph, name)
    , transform_dense_(graph, name + "_transform_dense",
                       config.hidden_size,
                       config.hidden_size,
                       true,
                       dtype)
    , transform_act_(graph, name + "_transform_act",
                     activation_type_from_config(config))
    , transform_ln_(graph, name + "_transform_ln",
                    config.hidden_size, -1, config.layer_norm_eps, 0, dtype)
    , decoder_(graph, name + "_decoder",
               config.hidden_size,
               config.vocab_size,
               false,
               dtype)
    , head_bias_tensor_(graph->tensor({config.vocab_size}, dtype, true))
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    head_bias_tensor_->set_name(tensor_name("head_bias"));
    register_parameter("head_bias", head_bias_tensor_);
    register_module("transform_dense", &transform_dense_);
    register_module("transform_act", &transform_act_);
    register_module("transform_ln", &transform_ln_);
    register_module("decoder", &decoder_);
}

NNGraph::TensorNode* RobertaMlmHead::forward(
    NNGraph::TensorNode* hidden)
{
    NNGraph::TensorNode* t = transform_dense_.forward(hidden);
    t = transform_act_.forward(t);
    t = transform_ln_.forward(t);

    NNGraph::TensorNode* logits = decoder_.forward(t);
    const Index feature_axis = logits->ndim() - 1;
    logits = add_fiber(1.0, head_bias_tensor_, 1.0, logits, feature_axis, 0);
    return logits;
}

std::string RobertaMlmHead::repr() const
{
    return "RobertaMlmHead(vocab=" + std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::roberta
