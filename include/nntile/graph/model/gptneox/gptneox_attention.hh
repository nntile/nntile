/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneox/gptneox_attention.hh
 * GPT-NeoX attention - self-attention with RoPE and SDPA.
 *
 * Input layout: (hidden_size, seq, batch) in Fortran order.
 * GPT-NeoX uses full attention heads (no GQA).
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/model/gptneox/gptneox_config.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::model::gptneox
{

//! GPT-NeoXAttention - Q/K/V projections via gemm, RoPE, SDPA, output projection
class GptneoxAttention : public graph::module::Module
{
private:
    graph::NNGraph::TensorNode* w_q_ = nullptr;  // (n_heads, head_size, n_emb)
    graph::NNGraph::TensorNode* w_k_ = nullptr;  // (n_heads, head_size, n_emb)
    graph::NNGraph::TensorNode* w_v_ = nullptr;  // (n_heads, head_size, n_emb)
    graph::NNGraph::TensorNode* w_o_ = nullptr;  // (n_emb, n_heads, head_size)

    GptneoxConfig config_;
    graph::DataType dtype_;

    Index head_size_;
    Index n_heads_;

public:
    //! Constructor
    GptneoxAttention(graph::NNGraph* graph,
                     const std::string& name,
                     const GptneoxConfig& config,
                     graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* sin = nullptr,
        graph::NNGraph::TensorNode* cos = nullptr,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    Index head_size() const { return head_size_; }
    Index num_heads() const { return n_heads_; }
};

} // namespace nntile::model::gptneox
