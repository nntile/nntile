/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/bert/bert_model.hh
 * BertModel - embeddings + encoder stack.
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nntile/graph.hh>
#include <nntile/model/bert/bert_config.hh>
#include <nntile/model/bert/bert_embeddings.hh>
#include <nntile/model/bert/bert_layer.hh>
#include <nntile/module/module.hh>

namespace nntile::model::bert
{

class BertModel : public module::Module
{
private:
    BertEmbeddings embeddings_;
    std::vector<std::unique_ptr<BertLayer>> layers_;

    BertConfig config_;
    DataType dtype_;

public:
    BertModel(NNGraph* graph,
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

    Index num_layers() const { return config_.num_hidden_layers; }

    NNGraph::TensorNode* word_vocab_tensor() const
    {
        return embeddings_.word_vocab_tensor();
    }
};

} // namespace nntile::model::bert
