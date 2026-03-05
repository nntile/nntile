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
 * Input layout: (seq, batch, hidden_size)
 * Uses sdpa_eager for scaled dot-product attention.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/llama/llama_config.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/module.hh>

namespace nntile::model::llama
{

//! LlamaAttention - Q/K/V projections, RoPE, SDPA, output projection
//! Supports num_attention_heads == num_key_value_heads (no GQA) for simplicity.
class LlamaAttention : public module::Module
{
private:
    module::Linear q_proj_;
    module::Linear k_proj_;
    module::Linear v_proj_;
    module::Linear out_proj_;

    LlamaConfig config_;
    graph::DataType dtype_;

    Index head_size_;
    Index n_heads_;
    Index n_head_kv_;
    Index kv_group_size_;

public:
    //! Constructor
    //! @param graph Pointer to the neural network graph
    //! @param name Module name
    //! @param config Llama configuration
    //! @param dtype Data type
    LlamaAttention(graph::NNGraph* graph,
                   const std::string& name,
                   const LlamaConfig& config,
                   graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    //! @param x Input tensor (seq, batch, hidden_size)
    //! @param sin RoPE sin tensor (head_size/2, seq, batch), may be nullptr to skip RoPE
    //! @param cos RoPE cos tensor (head_size/2, seq, batch), may be nullptr to skip RoPE
    //! @param mask Optional attention mask (k_seq, q_seq), may be nullptr
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* sin = nullptr,
        graph::NNGraph::TensorNode* cos = nullptr,
        graph::NNGraph::TensorNode* mask = nullptr);

    //! Get string representation
    std::string repr() const override;

    // Dimension accessors
    Index head_size() const { return head_size_; }
    Index num_heads() const { return n_heads_; }
};

} // namespace nntile::model::llama
