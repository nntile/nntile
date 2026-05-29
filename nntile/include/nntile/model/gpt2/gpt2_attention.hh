/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/gpt2/gpt2_attention.hh
 * GPT2Attention - self-attention with optional mask (no RoPE).
 *
 * GPT-2 uses combined c_attn (Q,K,V) and c_proj. We use separate Q/K/V/O
 * projections like LLaMA for consistency. Layout: (hidden_size, seq, batch).
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/gpt2/gpt2_config.hh>
#include <nntile/module/module.hh>

namespace nntile::model::gpt2
{

//! GPT2Attention - Q/K/V projections, SDPA, output projection
//! No RoPE, no GQA (num_attention_heads == num_key_value_heads)
class Gpt2Attention : public module::Module
{
private:
    NNGraph::TensorNode* w_q_ = nullptr;
    NNGraph::TensorNode* w_k_ = nullptr;
    NNGraph::TensorNode* w_v_ = nullptr;
    NNGraph::TensorNode* w_o_ = nullptr;
    NNGraph::TensorNode* q_bias_ = nullptr;
    NNGraph::TensorNode* k_bias_ = nullptr;
    NNGraph::TensorNode* v_bias_ = nullptr;
    NNGraph::TensorNode* o_bias_ = nullptr;

    Gpt2Config config_;
    DataType dtype_;

    Index head_size_;
    Index n_heads_;

public:
    //! Constructor
    Gpt2Attention(NNGraph* graph,
                  const std::string& name,
                  const Gpt2Config& config,
                  DataType dtype = DataType::FP32);

    //! Forward pass
    //! @param mask Optional BOOL mask for ``sdpa_eager``; ``nullptr`` is full
    //!        bidirectional attention.
    //! @param causal Placeholder (not implemented); must be ``false``.
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* x,
        NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;

    Index head_size() const { return head_size_; }
    Index num_heads() const { return n_heads_; }
};

} // namespace nntile::model::gpt2
