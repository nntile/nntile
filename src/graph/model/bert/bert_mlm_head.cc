/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/model/bert/bert_mlm_head.cc
 * BertMlmHead implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/bert/bert_mlm_head.hh"

namespace nntile::model::bert
{

BertMlmHead::BertMlmHead(graph::NNGraph* graph,
                         const std::string& name,
                         const BertConfig& config,
                         graph::DataType dtype)
    : graph::module::Module(graph, name)
    , transform_dense_(graph, name + "_transform_dense",
                       config.hidden_size,
                       config.hidden_size,
                       true,
                       dtype)
    , transform_act_(graph, name + "_transform_act",
                     graph::module::ActivationType::GELUTANH,
                     dtype)
    , transform_ln_(graph, name + "_transform_ln",
                    config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
    , decoder_(graph, name + "_decoder",
               config.hidden_size,
               config.vocab_size,
               true,
               dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("transform_dense", &transform_dense_);
    register_module("transform_act", &transform_act_);
    register_module("transform_ln", &transform_ln_);
    register_module("decoder", &decoder_);
}

graph::NNGraph::TensorNode* BertMlmHead::forward(
    graph::NNGraph::TensorNode* hidden)
{
    graph::NNGraph::TensorNode* t = transform_dense_.forward(hidden);
    t = transform_act_.forward(t);
    t = transform_ln_.forward(t);
    return decoder_.forward(t);
}

std::string BertMlmHead::repr() const
{
    return "BertMlmHead(vocab=" + std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::bert
