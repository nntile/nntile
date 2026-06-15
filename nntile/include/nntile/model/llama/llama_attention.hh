/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/llama/llama_attention.hh
 * LlamaAttention - self-attention with RoPE and sdpa_eager.
 *
 * Input layout: (batch, seq, hidden_size) in graph.
 * Mimics wrappers/python/nntile/model/llama_attention.py::forward_async():
 * - Q/K/V via gemm with 3D/4D weight matrices (not Linear)
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/llama/llama_config.hh>
#include <nntile/module/module.hh>

namespace nntile::model::llama
{

//! LlamaAttention - Q/K/V projections via gemm, RoPE, SDPA, output projection
//! Uses gemm directly (not Linear) to support 3D/4D weight layouts like Python.
class LlamaAttention : public module::Module
{
private:
    // Weight tensors: 3D/4D as in Python (not 2D Linear)
    NNGraph::TensorNode* w_q_ = nullptr;  // (n_emb, head_size, n_head_kv, kv_group_size) or (n_emb, head_size, n_heads)
    NNGraph::TensorNode* w_k_ = nullptr;  // (n_emb, head_size, n_head_kv)
    NNGraph::TensorNode* w_v_ = nullptr;  // (n_emb, head_size, n_head_kv)
    NNGraph::TensorNode* w_o_ = nullptr;   // (head_size, n_head_kv, kv_group_size, n_emb) or (head_size, n_heads, n_emb)

    LlamaConfig config_;
    DataType dtype_;

    Index head_size_;
    Index n_heads_;
    Index n_head_kv_;
    Index kv_group_size_;
    bool use_gqa_;  // true if n_head_kv < n_heads

public:
    //! Constructor
    //! @param graph Pointer to the neural network graph
    //! @param name Module name
    //! @param config Llama configuration
    //! @param dtype Data type
    LlamaAttention(NNGraph* graph,
                   const std::string& name,
                   const LlamaConfig& config,
                   DataType dtype = DataType::FP32);

    //! Forward pass
    //! @param x Input tensor (batch, seq, hidden_size)
    //! @param sin RoPE sin tensor (batch, seq, head_size/2), may be nullptr to
    //! skip RoPE. Buffers can be filled with ``rope_sin_cos_from_position_ids``
    //! (see ``llama_rope.hh``) from ``(batch, seq)`` position ids like HF.
    //! @param cos RoPE cos tensor (batch, seq, head_size/2), may be nullptr to
    //! skip RoPE
    //! @param mask Optional BOOL attention mask (k_seq, q_seq), may be
    //! nullptr. Build with ``nntile::sdpa_causal_mask_bool_fill``
    //! for causal
    //! LM.
    //! @param k_cache Optional KV cache for K
    //!     (n_head_kv, batch, max_seq, head_size) virtual graph
    //! @param v_cache Optional KV cache for V
    //!     (n_head_kv, batch, max_seq, head_size) virtual graph
    //! @param cache_len Current valid length in cache (0 = prefill, >0 = decode)
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* x,
        NNGraph::TensorNode* sin = nullptr,
        NNGraph::TensorNode* cos = nullptr,
        NNGraph::TensorNode* mask = nullptr,
        NNGraph::TensorNode* k_cache = nullptr,
        NNGraph::TensorNode* v_cache = nullptr,
        Index cache_len = 0);

    //! Get string representation
    std::string repr() const override;

    // Dimension accessors
    Index head_size() const { return head_size_; }
    Index num_heads() const { return n_heads_; }
};

} // namespace nntile::model::llama
