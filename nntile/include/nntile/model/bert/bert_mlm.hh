/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/bert/bert_mlm.hh
 * BertMlm - BertModel + MLM head (BertForMaskedLM).
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <string>

#include <nntile/graph.hh>
#include <nntile/model/bert/bert_config.hh>
#include <nntile/model/bert/bert_model.hh>
#include <nntile/model/bert/bert_mlm_head.hh>
#include <nntile/module/module.hh>

namespace nntile::model::bert
{

class BertMlm : public module::Module
{
private:
    std::unique_ptr<BertModel> model_;
    BertMlmHead cls_;

    BertConfig config_;
    DataType dtype_;

public:
    BertMlm(NNGraph* graph,
            const std::string& name,
            const BertConfig& config,
            DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input_ids,
        NNGraph::TensorNode* token_type_ids,
        NNGraph::TensorNode* position_ids,
        NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;

    BertModel* model() { return model_.get(); }
};

} // namespace nntile::model::bert
