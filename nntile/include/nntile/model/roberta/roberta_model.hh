/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/roberta/roberta_model.hh
 * RobertaModel - embeddings + encoder stack.
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nntile/graph.hh>
#include <nntile/model/bert/bert_layer.hh>
#include <nntile/model/roberta/roberta_config.hh>
#include <nntile/model/roberta/roberta_embeddings.hh>
#include <nntile/module/module.hh>

namespace nntile::model::roberta
{

class RobertaModel : public module::Module
{
private:
    RobertaEmbeddings embeddings_;
    std::vector<std::unique_ptr<bert::BertLayer>> layers_;

    RobertaConfig config_;
    DataType dtype_;

public:
    RobertaModel(NNGraph* graph,
                 const std::string& name,
                 const RobertaConfig& config,
                 DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input_ids,
        NNGraph::TensorNode* position_ids,
        NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;

    Index num_layers() const { return config_.num_hidden_layers; }

    NNGraph::TensorNode* word_vocab_tensor() const
    {
        return embeddings_.word_vocab_tensor();
    }
};

} // namespace nntile::model::roberta
