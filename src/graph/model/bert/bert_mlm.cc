/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/model/bert/bert_mlm.cc
 * BertMlm implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/bert/bert_mlm.hh"

#include <stdexcept>

namespace nntile::model::bert
{

BertMlm::BertMlm(graph::NNGraph* graph,
                 const std::string& name,
                 const BertConfig& config,
                 graph::DataType dtype)
    : graph::module::Module(graph, name)
    , model_(std::make_unique<BertModel>(graph, name + "_bert", config, dtype))
    , cls_(graph,
           name + "_cls",
           config,
           model_->word_vocab_tensor(),
           dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("bert", model_.get());
    register_module("cls", &cls_);
}

graph::NNGraph::TensorNode* BertMlm::forward(
    graph::NNGraph::TensorNode* input_ids,
    graph::NNGraph::TensorNode* token_type_ids,
    graph::NNGraph::TensorNode* position_ids,
    graph::NNGraph::TensorNode* mask,
    bool causal)
{
    if(input_ids == nullptr || token_type_ids == nullptr ||
       position_ids == nullptr)
    {
        throw std::invalid_argument(
            "BertMlm::forward: input_ids, token_type_ids, and position_ids "
            "must be non-null");
    }

    graph::NNGraph::TensorNode* hidden = model_->forward(
        input_ids, token_type_ids, position_ids, mask, causal);
    graph::NNGraph::TensorNode* logits = cls_.forward(hidden);
    logits->set_name(tensor_name("logits"));
    return logits;
}

std::string BertMlm::repr() const
{
    return "BertMlm(" + model_->repr() + ", vocab=" +
           std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::bert
