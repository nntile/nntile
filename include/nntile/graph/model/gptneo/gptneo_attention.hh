/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneo/gptneo_attention.hh
 * GPT-Neo attention - self-attention with causal mask, output projection bias.
 *
 * Input layout: (hidden_size, seq, batch) in Fortran order.
 * Uses standard multi-head attention: Q/K/V projections, SDPA, output projection.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/model/gptneo/gptneo_config.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::model::gptneo
{

//! GPT-Neo attention - Q/K/V projections, SDPA, output projection with bias
class GptneoAttention : public graph::module::Module
{
private:
    graph::NNGraph::TensorNode* w_q_ = nullptr;  // (n_heads, head_size, n_emb)
    graph::NNGraph::TensorNode* w_k_ = nullptr;
    graph::NNGraph::TensorNode* w_v_ = nullptr;
    graph::NNGraph::TensorNode* w_o_ = nullptr;  // (n_emb, n_heads, head_size)
    graph::NNGraph::TensorNode* out_bias_ = nullptr;  // (n_emb)

    GptneoConfig config_;
    graph::DataType dtype_;

    Index head_size_;
    Index n_heads_;

public:
    //! Constructor
    GptneoAttention(graph::NNGraph* graph,
                    const std::string& name,
                    const GptneoConfig& config,
                    graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    Index head_size() const { return head_size_; }
    Index num_heads() const { return n_heads_; }
};

} // namespace nntile::model::gptneo
