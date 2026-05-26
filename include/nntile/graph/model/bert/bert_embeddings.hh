/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/bert/bert_embeddings.hh
 * BertEmbeddings - word, position, token-type embeddings + LayerNorm.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/graph/model/bert/bert_config.hh>
#include <nntile/graph/module/embedding.hh>
#include <nntile/graph/module/layer_norm.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::model::bert
{

class BertEmbeddings : public graph::module::Module
{
private:
    graph::module::Embedding word_embeddings_;
    graph::module::Embedding position_embeddings_;
    graph::module::Embedding token_type_embeddings_;
    graph::module::LayerNorm layer_norm_;

    BertConfig config_;
    graph::DataType dtype_;

public:
    BertEmbeddings(graph::NNGraph* graph,
                   const std::string& name,
                   const BertConfig& config,
                   graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input_ids,
        graph::NNGraph::TensorNode* token_type_ids,
        graph::NNGraph::TensorNode* position_ids);

    std::string repr() const override;

    graph::NNGraph::TensorNode* word_vocab_tensor() const
    {
        return word_embeddings_.vocab_tensor();
    }
};

} // namespace nntile::model::bert
