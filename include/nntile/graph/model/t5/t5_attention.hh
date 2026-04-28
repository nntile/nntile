/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/t5/t5_attention.hh
 * T5Attention - self-attention or cross-attention (no RoPE, no relative bias).
 *
 * Input layout: (d_model, seq, batch) in Fortran order.
 * T5 uses scaled dot-product attention with 1/sqrt(d_k).
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/model/t5/t5_config.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::model::t5
{

//! T5Attention - Q/K/V projections via gemm, SDPA, output projection
//! For self-attention: Q,K,V from same input x
//! For cross-attention: Q from x, K,V from encoder_output
class T5Attention : public graph::module::Module
{
private:
    graph::NNGraph::TensorNode* w_q_ = nullptr;  // (n_heads, head_size, d_model)
    graph::NNGraph::TensorNode* w_k_ = nullptr;  // (n_heads, head_size, d_kv) for cross, d_model for self
    graph::NNGraph::TensorNode* w_v_ = nullptr;  // (n_heads, head_size, d_kv) for cross, d_model for self
    graph::NNGraph::TensorNode* w_o_ = nullptr;  // (d_model, n_heads, head_size)

    T5Config config_;
    graph::DataType dtype_;
    bool is_cross_attention_;

    Index head_size_;
    Index n_heads_;

public:
    //! Constructor
    T5Attention(graph::NNGraph* graph,
                const std::string& name,
                const T5Config& config,
                bool is_cross_attention = false,
                graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    //! @param x Input (d_model, seq, batch) - query source
    //! @param encoder_output For cross-attention: (d_model, enc_seq, batch). For self-attn: nullptr
    //! @param mask Optional attention mask (k_seq, q_seq)
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* encoder_output = nullptr,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    Index head_size() const { return head_size_; }
    Index num_heads() const { return n_heads_; }
    bool is_cross_attention() const { return is_cross_attention_; }
};

} // namespace nntile::model::t5
