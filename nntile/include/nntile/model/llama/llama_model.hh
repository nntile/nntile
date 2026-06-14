/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/llama/llama_model.hh
 * LlamaModel - embedding + decoder layers + final norm.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <string>
#include <utility>
#include <vector>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/kv_cache.hh>
#include <nntile/model/llama/llama_config.hh>
#include <nntile/model/llama/llama_decoder.hh>
#include <nntile/module/embedding.hh>
#include <nntile/module/rms_norm.hh>
#include <nntile/module/module.hh>

namespace nntile::model::llama
{

//! LlamaModel - embed_tokens + num_hidden_layers x LlamaDecoder + norm
class LlamaModel : public module::Module
{
private:
    module::Embedding embed_tokens_;
    std::vector<std::unique_ptr<LlamaDecoder>> layers_;
    module::RMSNorm norm_;

    LlamaConfig config_;
    DataType dtype_;

public:
    //! Constructor
    LlamaModel(NNGraph* graph,
               const std::string& name,
               const LlamaConfig& config,
               DataType dtype = DataType::FP32);

    //! Forward pass
    //! @param input_ids (batch, seq) INT64 token indices; output is (batch, seq, hidden_size)
    //! @param sin RoPE sin per layer (optional)
    //! @param cos RoPE cos per layer (optional)
    //! @param mask Attention mask (optional)
    //! @param kv_cache Optional KV cache; when non-null, uses cache for decode
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input_ids,
        NNGraph::TensorNode* sin = nullptr,
        NNGraph::TensorNode* cos = nullptr,
        NNGraph::TensorNode* mask = nullptr,
        KVCache* kv_cache = nullptr);

    std::string repr() const override;

    Index num_layers() const { return config_.num_hidden_layers; }
};

} // namespace nntile::model::llama
