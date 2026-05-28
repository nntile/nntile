/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/model/roberta/roberta_mlm_head.cc
 * RobertaMlmHead implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/roberta/roberta_mlm_head.hh"
#include "nntile/graph/nn/ops/add_fiber.hh"
#include "nntile/graph/nn/ops/gemm.hh"

namespace nntile::graph::model::roberta
{

RobertaMlmHead::RobertaMlmHead(graph::NNGraph* graph,
                               const std::string& name,
                               const RobertaConfig& config,
                               graph::DataType dtype)
    : graph::module::Module(graph, name)
    , transform_dense_(graph, name + "_transform_dense",
                       config.hidden_size,
                       config.hidden_size,
                       true,
                       dtype)
    , transform_act_(graph, name + "_transform_act",
                     activation_type_from_config(config))
    , transform_ln_(graph, name + "_transform_ln",
                    config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
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

graph::NNGraph::TensorNode* RobertaMlmHead::forward(
    graph::NNGraph::TensorNode* hidden)
{
    graph::NNGraph::TensorNode* t = graph::gemm(
        transform_dense_.weight_tensor(),
        hidden,
        1.0,
        true,
        false,
        1,
        0);
    t = graph::add_fiber(
        1.0, transform_dense_.bias_tensor(), 1.0, t, 0, 0);
    t = transform_act_.forward(t);
    t = transform_ln_.forward(t);

    graph::NNGraph::TensorNode* logits = graph::gemm(
        decoder_.weight_tensor(),
        t,
        1.0,
        true,
        false,
        1,
        0);
    logits = graph::add_fiber(1.0, head_bias_tensor_, 1.0, logits, 0, 0);
    return logits;
}

std::string RobertaMlmHead::repr() const
{
    return "RobertaMlmHead(vocab=" + std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::graph::model::roberta
