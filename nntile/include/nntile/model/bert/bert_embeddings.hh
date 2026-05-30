/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/bert/bert_embeddings.hh
 * BertEmbeddings - word, position, token-type embeddings + LayerNorm.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/model/bert/bert_config.hh>
#include <nntile/module/embedding.hh>
#include <nntile/module/layer_norm.hh>
#include <nntile/module/module.hh>

namespace nntile::model::bert
{

class BertEmbeddings : public module::Module
{
private:
    module::Embedding word_embeddings_;
    module::Embedding position_embeddings_;
    module::Embedding token_type_embeddings_;
    module::LayerNorm layer_norm_;

    BertConfig config_;
    DataType dtype_;

public:
    BertEmbeddings(NNGraph* graph,
                   const std::string& name,
                   const BertConfig& config,
                   DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input_ids,
        NNGraph::TensorNode* token_type_ids,
        NNGraph::TensorNode* position_ids);

    std::string repr() const override;

    NNGraph::TensorNode* word_vocab_tensor() const
    {
        return word_embeddings_.vocab_tensor();
    }
};

} // namespace nntile::model::bert
