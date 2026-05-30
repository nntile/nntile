/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/gptneox/gptneox_model.hh
 * GptneoxModel - embedding + decoder layers + final norm.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <string>
#include <vector>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/gptneox/gptneox_config.hh>
#include <nntile/model/gptneox/gptneox_decoder.hh>
#include <nntile/module/embedding.hh>
#include <nntile/module/module.hh>
#include <nntile/module/layer_norm.hh>

namespace nntile::model::gptneox
{

//! GptneoxModel - embed_tokens + num_hidden_layers x GptneoxDecoder + norm
class GptneoxModel : public module::Module
{
private:
    module::Embedding embed_tokens_;
    std::vector<std::unique_ptr<GptneoxDecoder>> layers_;
    module::LayerNorm norm_;

    GptneoxConfig config_;
    DataType dtype_;

public:
    //! Constructor
    GptneoxModel(NNGraph* graph,
                 const std::string& name,
                 const GptneoxConfig& config,
                 DataType dtype = DataType::FP32);

    //! Forward pass
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input_ids,
        NNGraph::TensorNode* sin = nullptr,
        NNGraph::TensorNode* cos = nullptr,
        NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    Index num_layers() const { return config_.num_hidden_layers; }

    NNGraph::TensorNode* embed_vocab_tensor() const
    {
        return embed_tokens_.vocab_tensor();
    }
};

} // namespace nntile::model::gptneox
