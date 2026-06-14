/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/roberta/roberta_embeddings.cc
 * RobertaEmbeddings implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/roberta/roberta_embeddings.hh"
#include "nntile/nn/ops/add.hh"
#include "nntile/nn/ops/transpose.hh"

#include <stdexcept>

namespace nntile::model::roberta
{

RobertaEmbeddings::RobertaEmbeddings(NNGraph* graph,
                                     const std::string& name,
                                     const RobertaConfig& config,
                                     DataType dtype)
    : module::Module(graph, name)
    , word_embeddings_(graph, name + "_word",
                       config.vocab_size, config.hidden_size,
                       2, 0, dtype)
    , position_embeddings_(graph, name + "_position",
                           config.max_position_embeddings,
                           config.hidden_size,
                           2, 0, dtype)
    , layer_norm_(graph, name + "_ln",
                  config.hidden_size, 2, config.layer_norm_eps, 0, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("word", &word_embeddings_);
    register_module("position", &position_embeddings_);
    register_module("ln", &layer_norm_);
}

NNGraph::TensorNode* RobertaEmbeddings::forward(
    NNGraph::TensorNode* input_ids,
    NNGraph::TensorNode* position_ids)
{
    if(input_ids == nullptr || position_ids == nullptr)
    {
        throw std::invalid_argument(
            "RobertaEmbeddings::forward: input_ids and position_ids must be "
            "non-null");
    }

    NNGraph::TensorNode* word =
        word_embeddings_.forward(input_ids);
    NNGraph::TensorNode* position =
        position_embeddings_.forward(position_ids);

    NNGraph::TensorNode* embed =
        add(1.0, word, 1.0, position);
    NNGraph::TensorNode* x = transpose(embed, 2);
    return layer_norm_.forward(x);
}

std::string RobertaEmbeddings::repr() const
{
    return "RobertaEmbeddings(hidden=" +
           std::to_string(config_.hidden_size) +
           ", vocab=" + std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::roberta
