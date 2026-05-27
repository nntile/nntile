/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/roberta/roberta_embeddings.hh
 * RobertaEmbeddings - word and position embeddings + LayerNorm.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/graph/model/roberta/roberta_config.hh>
#include <nntile/graph/module/embedding.hh>
#include <nntile/graph/module/layer_norm.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::graph::model::roberta
{

class RobertaEmbeddings : public graph::module::Module
{
private:
    graph::module::Embedding word_embeddings_;
    graph::module::Embedding position_embeddings_;
    graph::module::LayerNorm layer_norm_;

    RobertaConfig config_;
    graph::DataType dtype_;

public:
    RobertaEmbeddings(graph::NNGraph* graph,
                      const std::string& name,
                      const RobertaConfig& config,
                      graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input_ids,
        graph::NNGraph::TensorNode* position_ids);

    std::string repr() const override;

    graph::NNGraph::TensorNode* word_vocab_tensor() const
    {
        return word_embeddings_.vocab_tensor();
    }
};

} // namespace nntile::graph::model::roberta
