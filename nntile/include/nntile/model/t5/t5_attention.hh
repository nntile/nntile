/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/t5/t5_attention.hh
 * T5Attention - self-attention or cross-attention (no RoPE, no relative bias).
 *
 * Input layout: (batch, seq, d_model) in C-order.
 * T5 uses scaled dot-product attention with 1/sqrt(d_k).
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/t5/t5_config.hh>
#include <nntile/module/module.hh>

namespace nntile::model::t5
{

//! T5Attention - Q/K/V projections via gemm, SDPA, output projection
//! For self-attention: Q,K,V from same input x
//! For cross-attention: Q from x, K,V from encoder_output
class T5Attention : public module::Module
{
private:
    NNGraph::TensorNode* w_q_ = nullptr;  // (n_heads, head_size, d_model)
    NNGraph::TensorNode* w_k_ = nullptr;  // (n_heads, head_size, d_kv) for cross, d_model for self
    NNGraph::TensorNode* w_v_ = nullptr;  // (n_heads, head_size, d_kv) for cross, d_model for self
    NNGraph::TensorNode* w_o_ = nullptr;  // (d_model, n_heads, head_size)

    T5Config config_;
    DataType dtype_;
    bool is_cross_attention_;

    Index head_size_;
    Index n_heads_;

public:
    //! Constructor
    T5Attention(NNGraph* graph,
                const std::string& name,
                const T5Config& config,
                bool is_cross_attention = false,
                DataType dtype = DataType::FP32);

    //! Forward pass
    //! @param x Input (batch, seq, d_model) - query source
    //! @param encoder_output For cross-attention: (batch, enc_seq, d_model). For self-attn: nullptr
    //! @param mask Optional attention mask (k_seq, q_seq)
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* x,
        NNGraph::TensorNode* encoder_output = nullptr,
        NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    Index head_size() const { return head_size_; }
    Index num_heads() const { return n_heads_; }
    bool is_cross_attention() const { return is_cross_attention_; }
};

} // namespace nntile::model::t5
