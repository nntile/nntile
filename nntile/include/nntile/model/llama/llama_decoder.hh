/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/llama/llama_decoder.hh
 * LlamaDecoder - one transformer block (attention + MLP with residuals).
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/llama/llama_attention.hh>
#include <nntile/model/llama/llama_config.hh>
#include <nntile/model/llama/llama_mlp.hh>
#include <nntile/module/module.hh>
#include <nntile/module/rms_norm.hh>

namespace nntile::model::llama
{

//! LlamaDecoder - input_norm -> attention -> residual -> post_attn_norm -> MLP -> residual
class LlamaDecoder : public module::Module
{
private:
    module::RMSNorm input_norm_;
    LlamaAttention attention_;
    module::RMSNorm post_attn_norm_;
    LlamaMLP mlp_;

    LlamaConfig config_;
    DataType dtype_;

public:
    //! Constructor
    LlamaDecoder(NNGraph* graph,
                 const std::string& name,
                 const LlamaConfig& config,
                 DataType dtype = DataType::FP32);

    //! Forward pass
    //! @param x Input (hidden_size, seq, batch) in Fortran order
    //! @param sin RoPE sin (optional)
    //! @param cos RoPE cos (optional)
    //! @param mask Attention mask (optional)
    //! @param k_cache Optional KV cache for K (optional)
    //! @param v_cache Optional KV cache for V (optional)
    //! @param cache_len Current valid length in cache (0 = prefill)
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* x,
        NNGraph::TensorNode* sin = nullptr,
        NNGraph::TensorNode* cos = nullptr,
        NNGraph::TensorNode* mask = nullptr,
        NNGraph::TensorNode* k_cache = nullptr,
        NNGraph::TensorNode* v_cache = nullptr,
        Index cache_len = 0);

    std::string repr() const override;

    LlamaAttention& attention() { return attention_; }
    LlamaMLP& mlp() { return mlp_; }
};

} // namespace nntile::model::llama
