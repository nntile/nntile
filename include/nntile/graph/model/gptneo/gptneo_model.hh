/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneo/gptneo_model.hh
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
#include <nntile/graph/model/gptneo/gptneo_config.hh>
#include <nntile/graph/model/gptneo/gptneo_decoder.hh>
#include <nntile/graph/module/embedding.hh>
#include <nntile/graph/module/module.hh>
#include <nntile/graph/module/rms_norm.hh>

namespace nntile::model::gptneo
{

//! GPT-Neo model - wte + wpe + add + num_hidden_layers x GptneoDecoder + norm
class GptneoModel : public graph::module::Module
{
private:
    graph::module::Embedding wte_;
    graph::module::Embedding wpe_;
    std::vector<std::unique_ptr<GptneoDecoder>> layers_;
    graph::module::RMSNorm norm_;

    GptneoConfig config_;
    graph::DataType dtype_;

public:
    //! Constructor
    GptneoModel(graph::NNGraph* graph,
                const std::string& name,
                const GptneoConfig& config,
                graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    //! @param input_ids (seq, batch) INT64 token indices
    //! @param position_ids (seq, batch) INT64 position indices (optional; if null, uses arange)
    //! @param mask Attention mask (optional)
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input_ids,
        graph::NNGraph::TensorNode* position_ids = nullptr,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    Index num_layers() const { return config_.num_hidden_layers; }
};

} // namespace nntile::model::gptneo
