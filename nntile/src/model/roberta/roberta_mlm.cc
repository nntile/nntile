/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/model/roberta/roberta_mlm.cc
 * RobertaMlm implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/roberta/roberta_mlm.hh"

#include <stdexcept>

namespace nntile::model::roberta
{

RobertaMlm::RobertaMlm(NNGraph* graph,
                       const std::string& name,
                       const RobertaConfig& config,
                       DataType dtype)
    : module::Module(graph, name)
    , model_(std::make_unique<RobertaModel>(
          graph, name + "_roberta", config, dtype))
    , cls_(graph, name + "_cls", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("roberta", model_.get());
    register_module("cls", &cls_);
}

NNGraph::TensorNode* RobertaMlm::forward(
    NNGraph::TensorNode* input_ids,
    NNGraph::TensorNode* position_ids,
    NNGraph::TensorNode* mask,
    bool causal)
{
    if(input_ids == nullptr || position_ids == nullptr)
    {
        throw std::invalid_argument(
            "RobertaMlm::forward: input_ids and position_ids must be "
            "non-null");
    }

    NNGraph::TensorNode* hidden =
        model_->forward(input_ids, position_ids, mask, causal);
    NNGraph::TensorNode* logits = cls_.forward(hidden);
    logits->set_name(tensor_name("logits"));
    return logits;
}

std::string RobertaMlm::repr() const
{
    return "RobertaMlm(" + model_->repr() + ", vocab=" +
           std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::roberta
