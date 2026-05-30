/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/roberta/roberta_embeddings.hh
 * RobertaEmbeddings - word and position embeddings + LayerNorm.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/model/roberta/roberta_config.hh>
#include <nntile/module/embedding.hh>
#include <nntile/module/layer_norm.hh>
#include <nntile/module/module.hh>

namespace nntile::model::roberta
{

class RobertaEmbeddings : public module::Module
{
private:
    module::Embedding word_embeddings_;
    module::Embedding position_embeddings_;
    module::LayerNorm layer_norm_;

    RobertaConfig config_;
    DataType dtype_;

public:
    RobertaEmbeddings(NNGraph* graph,
                      const std::string& name,
                      const RobertaConfig& config,
                      DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input_ids,
        NNGraph::TensorNode* position_ids);

    std::string repr() const override;

    NNGraph::TensorNode* word_vocab_tensor() const
    {
        return word_embeddings_.vocab_tensor();
    }
};

} // namespace nntile::model::roberta
