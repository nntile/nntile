/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/gptneox/gptneox_attention.hh
 * GPT-NeoX attention - self-attention with RoPE and SDPA.
 *
 * Input layout: (batch, seq, hidden_size) in C-order.
 * GPT-NeoX uses full attention heads (no GQA).
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/gptneox/gptneox_config.hh>
#include <nntile/module/module.hh>

namespace nntile::model::gptneox
{

//! GPT-NeoXAttention - Q/K/V projections via gemm, RoPE, SDPA, output projection
class GptneoxAttention : public module::Module
{
private:
    NNGraph::TensorNode* w_q_ = nullptr;  // (n_heads, head_size, n_emb)
    NNGraph::TensorNode* w_k_ = nullptr;  // (n_heads, head_size, n_emb)
    NNGraph::TensorNode* w_v_ = nullptr;  // (n_heads, head_size, n_emb)
    NNGraph::TensorNode* w_o_ = nullptr;  // (n_emb, n_heads, head_size)

    GptneoxConfig config_;
    DataType dtype_;

    Index head_size_;
    Index n_heads_;

public:
    //! Constructor
    GptneoxAttention(NNGraph* graph,
                     const std::string& name,
                     const GptneoxConfig& config,
                     DataType dtype = DataType::FP32);

    //! Forward pass
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* x,
        NNGraph::TensorNode* sin = nullptr,
        NNGraph::TensorNode* cos = nullptr,
        NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    Index head_size() const { return head_size_; }
    Index num_heads() const { return n_heads_; }
};

} // namespace nntile::model::gptneox
