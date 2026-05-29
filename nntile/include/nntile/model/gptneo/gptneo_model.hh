/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/gptneo/gptneo_model.hh
 * GPT-Neo model - wte + wpe + decoder layers + final norm.
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
#include <nntile/model/gptneo/gptneo_config.hh>
#include <nntile/model/gptneo/gptneo_decoder.hh>
#include <nntile/module/embedding.hh>
#include <nntile/module/module.hh>
#include <nntile/module/layer_norm.hh>

namespace nntile::model::gptneo
{

//! GPT-Neo model - wte + wpe + add + num_hidden_layers x GptneoDecoder + norm
class GptneoModel : public module::Module
{
private:
    module::Embedding wte_;
    module::Embedding wpe_;
    std::vector<std::unique_ptr<GptneoDecoder>> layers_;
    module::LayerNorm norm_;

    GptneoConfig config_;
    DataType dtype_;

public:
    //! Constructor
    GptneoModel(NNGraph* graph,
                const std::string& name,
                const GptneoConfig& config,
                DataType dtype = DataType::FP32);

    //! Forward pass
    //! @param input_ids (seq, batch) INT64 token indices
    //! @param position_ids (seq, batch) INT64 position indices (required)
    //! @param mask BOOL mask for global-attention layers (even ``layer_id``)
    //! @param local_mask BOOL mask for local-attention layers (odd ``layer_id``);
    //!        when null, ``mask`` is used for all layers (legacy / tests only)
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input_ids,
        NNGraph::TensorNode* position_ids,
        NNGraph::TensorNode* mask = nullptr,
        NNGraph::TensorNode* local_mask = nullptr);

    std::string repr() const override;

    Index num_layers() const { return config_.num_hidden_layers; }

    NNGraph::TensorNode* wte_vocab_tensor() const
    {
        return wte_.vocab_tensor();
    }

    NNGraph::TensorNode* wpe_vocab_tensor() const
    {
        return wpe_.vocab_tensor();
    }
};

} // namespace nntile::model::gptneo
