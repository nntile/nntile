/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/llama/llama_model.hh
 * LlamaModel - embedding + decoder layers + final norm.
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
#include <nntile/graph/model/llama/llama_config.hh>
#include <nntile/graph/model/llama/llama_decoder.hh>
#include <nntile/graph/module/embedding.hh>
#include <nntile/graph/module/rms_norm.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::model::llama
{

//! LlamaModel - embed_tokens + num_hidden_layers x LlamaDecoder + norm
class LlamaModel : public graph::module::Module
{
private:
    graph::module::Embedding embed_tokens_;
    std::vector<std::unique_ptr<LlamaDecoder>> layers_;
    graph::module::RMSNorm norm_;

    LlamaConfig config_;
    graph::DataType dtype_;

public:
    //! Constructor
    LlamaModel(graph::NNGraph* graph,
               const std::string& name,
               const LlamaConfig& config,
               graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    //! @param input_ids (seq, batch) INT64 token indices; output is (hidden_size, seq, batch)
    //! @param sin RoPE sin per layer (optional)
    //! @param cos RoPE cos per layer (optional)
    //! @param mask Attention mask (optional)
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input_ids,
        graph::NNGraph::TensorNode* sin = nullptr,
        graph::NNGraph::TensorNode* cos = nullptr,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    Index num_layers() const { return config_.num_hidden_layers; }
};

} // namespace nntile::model::llama
