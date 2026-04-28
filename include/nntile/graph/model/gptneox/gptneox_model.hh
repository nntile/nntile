/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneox/gptneox_model.hh
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
#include <nntile/graph/model/gptneox/gptneox_config.hh>
#include <nntile/graph/model/gptneox/gptneox_decoder.hh>
#include <nntile/graph/module/embedding.hh>
#include <nntile/graph/module/module.hh>
#include <nntile/graph/module/rms_norm.hh>

namespace nntile::model::gptneox
{

//! GptneoxModel - embed_tokens + num_hidden_layers x GptneoxDecoder + norm
class GptneoxModel : public graph::module::Module
{
private:
    graph::module::Embedding embed_tokens_;
    std::vector<std::unique_ptr<GptneoxDecoder>> layers_;
    graph::module::RMSNorm norm_;

    GptneoxConfig config_;
    graph::DataType dtype_;

public:
    //! Constructor
    GptneoxModel(graph::NNGraph* graph,
                 const std::string& name,
                 const GptneoxConfig& config,
                 graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input_ids,
        graph::NNGraph::TensorNode* sin = nullptr,
        graph::NNGraph::TensorNode* cos = nullptr,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    Index num_layers() const { return config_.num_hidden_layers; }
};

} // namespace nntile::model::gptneox
