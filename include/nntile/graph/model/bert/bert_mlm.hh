/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/bert/bert_mlm.hh
 * BertMlm - BertModel + MLM head (BertForMaskedLM).
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <string>

#include <nntile/graph.hh>
#include <nntile/graph/model/bert/bert_config.hh>
#include <nntile/graph/model/bert/bert_model.hh>
#include <nntile/graph/model/bert/bert_mlm_head.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::graph::model::bert
{

class BertMlm : public graph::module::Module
{
private:
    std::unique_ptr<BertModel> model_;
    BertMlmHead cls_;

    BertConfig config_;
    graph::DataType dtype_;

public:
    BertMlm(graph::NNGraph* graph,
            const std::string& name,
            const BertConfig& config,
            graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input_ids,
        graph::NNGraph::TensorNode* token_type_ids,
        graph::NNGraph::TensorNode* position_ids,
        graph::NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;

    BertModel* model() { return model_.get(); }
};

} // namespace nntile::graph::model::bert
