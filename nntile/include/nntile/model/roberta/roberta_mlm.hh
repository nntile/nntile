/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/roberta/roberta_mlm.hh
 * RobertaMlm - RobertaModel + MLM head (RobertaForMaskedLM).
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <string>

#include <nntile/graph.hh>
#include <nntile/model/roberta/roberta_config.hh>
#include <nntile/model/roberta/roberta_model.hh>
#include <nntile/model/roberta/roberta_mlm_head.hh>
#include <nntile/module/module.hh>

namespace nntile::model::roberta
{

class RobertaMlm : public module::Module
{
private:
    std::unique_ptr<RobertaModel> model_;
    RobertaMlmHead cls_;

    RobertaConfig config_;
    DataType dtype_;

public:
    RobertaMlm(NNGraph* graph,
               const std::string& name,
               const RobertaConfig& config,
               DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input_ids,
        NNGraph::TensorNode* position_ids,
        NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;

    RobertaModel* model() { return model_.get(); }
};

} // namespace nntile::model::roberta
