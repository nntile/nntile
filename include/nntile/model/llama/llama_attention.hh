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
 * Input layout: (hidden_size, seq, batch) in Fortran order.
 * Mimics wrappers/python/nntile/model/llama_attention.py::forward_async():
 * - No transpose on input before Q/K/V
 * - Q/K/V via gemm with 3D/4D weight matrices (not Linear)
 * - Transpose applied to Q, K, V outputs (not to input)
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/io/safetensors.hh>
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
    graph::NNGraph::TensorNode* w_q_ = nullptr;  // (kv_group_size, n_head_kv, head_size, n_emb) or (n_heads, head_size, n_emb)
    graph::NNGraph::TensorNode* w_k_ = nullptr;  // (n_head_kv, head_size, n_emb)
    graph::NNGraph::TensorNode* w_v_ = nullptr;  // (n_head_kv, head_size, n_emb)
    graph::NNGraph::TensorNode* w_o_ = nullptr;   // (n_emb, kv_group_size, n_head_kv, head_size) or (n_emb, n_heads, head_size)

    LlamaConfig config_;
    graph::DataType dtype_;

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
    LlamaAttention(graph::NNGraph* graph,
                   const std::string& name,
                   const LlamaConfig& config,
                   graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    //! @param x Input tensor (hidden_size, seq, batch)
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

    //! Import weights from HF-format SafeTensors (q_proj, k_proj, v_proj, o_proj)
    void import_hf(const io::SafeTensorsReader& reader,
                   const std::string& hf_prefix) override;

    //! Export weights to HF-format SafeTensors
    void export_hf(io::SafeTensorsWriter& writer,
                   const std::string& hf_prefix) const override;

    // Dimension accessors
    Index head_size() const { return head_size_; }
    Index num_heads() const { return n_heads_; }
};

} // namespace nntile::model::llama
