/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/model/bert/bert_embeddings.cc
 * BertEmbeddings implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/bert/bert_embeddings.hh"
#include "nntile/graph/nn/ops/add.hh"
#include "nntile/graph/nn/ops/transpose.hh"

#include <stdexcept>

namespace nntile::model::bert
{

BertEmbeddings::BertEmbeddings(graph::NNGraph* graph,
                               const std::string& name,
                               const BertConfig& config,
                               graph::DataType dtype)
    : graph::module::Module(graph, name)
    , word_embeddings_(graph, name + "_word",
                       config.vocab_size, config.hidden_size,
                       2, 0, dtype)
    , position_embeddings_(graph, name + "_position",
                           config.max_position_embeddings,
                           config.hidden_size,
                           2, 0, dtype)
    , token_type_embeddings_(graph, name + "_token_type",
                            config.type_vocab_size,
                            config.hidden_size,
                            2, 0, dtype)
    , layer_norm_(graph, name + "_ln",
                  config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("word", &word_embeddings_);
    register_module("position", &position_embeddings_);
    register_module("token_type", &token_type_embeddings_);
    register_module("ln", &layer_norm_);
}

graph::NNGraph::TensorNode* BertEmbeddings::forward(
    graph::NNGraph::TensorNode* input_ids,
    graph::NNGraph::TensorNode* token_type_ids,
    graph::NNGraph::TensorNode* position_ids)
{
    if(input_ids == nullptr || token_type_ids == nullptr ||
       position_ids == nullptr)
    {
        throw std::invalid_argument(
            "BertEmbeddings::forward: input_ids, token_type_ids, and "
            "position_ids must be non-null");
    }

    graph::NNGraph::TensorNode* word =
        word_embeddings_.forward(input_ids);
    graph::NNGraph::TensorNode* token_type =
        token_type_embeddings_.forward(token_type_ids);
    graph::NNGraph::TensorNode* position =
        position_embeddings_.forward(position_ids);

    graph::NNGraph::TensorNode* wt =
        graph::add(1.0, word, 1.0, token_type);
    graph::NNGraph::TensorNode* embed =
        graph::add(1.0, wt, 1.0, position);
    graph::NNGraph::TensorNode* x =
        graph::transpose(embed, 2);
    return layer_norm_.forward(x);
}

std::string BertEmbeddings::repr() const
{
    return "BertEmbeddings(hidden=" + std::to_string(config_.hidden_size) +
           ", vocab=" + std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::bert
