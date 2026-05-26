/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/roberta/roberta_mlm.hh
 * RobertaMlm - RobertaModel + MLM head (RobertaForMaskedLM).
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <string>

#include <nntile/graph.hh>
#include <nntile/graph/model/roberta/roberta_config.hh>
#include <nntile/graph/model/roberta/roberta_model.hh>
#include <nntile/graph/model/roberta/roberta_mlm_head.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::model::roberta
{

class RobertaMlm : public graph::module::Module
{
private:
    std::unique_ptr<RobertaModel> model_;
    RobertaMlmHead cls_;

    RobertaConfig config_;
    graph::DataType dtype_;

public:
    RobertaMlm(graph::NNGraph* graph,
               const std::string& name,
               const RobertaConfig& config,
               graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input_ids,
        graph::NNGraph::TensorNode* position_ids,
        graph::NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;

    RobertaModel* model() { return model_.get(); }
};

} // namespace nntile::model::roberta
