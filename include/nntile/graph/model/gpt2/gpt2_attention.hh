/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gpt2/gpt2_attention.hh
 * GPT2Attention - self-attention with causal mask (no RoPE).
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
#include <nntile/graph/model/gpt2/gpt2_config.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::model::gpt2
{

//! GPT2Attention - Q/K/V projections, SDPA with causal mask, output projection
//! No RoPE, no GQA (num_attention_heads == num_key_value_heads)
class Gpt2Attention : public graph::module::Module
{
private:
    graph::NNGraph::TensorNode* w_q_ = nullptr;
    graph::NNGraph::TensorNode* w_k_ = nullptr;
    graph::NNGraph::TensorNode* w_v_ = nullptr;
    graph::NNGraph::TensorNode* w_o_ = nullptr;

    Gpt2Config config_;
    graph::DataType dtype_;

    Index head_size_;
    Index n_heads_;

public:
    //! Constructor
    Gpt2Attention(graph::NNGraph* graph,
                  const std::string& name,
                  const Gpt2Config& config,
                  graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    Index head_size() const { return head_size_; }
    Index num_heads() const { return n_heads_; }
};

} // namespace nntile::model::gpt2
