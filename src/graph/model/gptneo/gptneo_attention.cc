/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gptneo/gptneo_attention.cc
 * GPT-Neo attention implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gptneo/gptneo_attention.hh"
#include "nntile/graph/nn/add_fiber.hh"
#include "nntile/graph/nn/gemm.hh"
#include "nntile/graph/nn/sdpa_eager.hh"
#include "nntile/graph/nn/transpose.hh"

#include <cmath>
#include <stdexcept>

namespace nntile::model::gptneo
{

GptneoAttention::GptneoAttention(graph::NNGraph* graph,
                                 const std::string& name,
                                 const GptneoConfig& config,
                                 graph::DataType dtype)
    : graph::module::Module(graph, name)
    , config_(config)
    , dtype_(dtype)
    , head_size_(config.head_dim)
    , n_heads_(config.num_attention_heads)
{
    Index n_emb = config.hidden_size;

    // w_q, w_k, w_v: (n_heads, head_size, n_emb)
    w_q_ = graph_->tensor(
        {n_heads_, head_size_, n_emb},
        tensor_name("q_weight"),
        dtype_,
        true);
    register_parameter("q_weight", w_q_);

    w_k_ = graph_->tensor(
        {n_heads_, head_size_, n_emb},
        tensor_name("k_weight"),
        dtype_,
        true);
    register_parameter("k_weight", w_k_);

    w_v_ = graph_->tensor(
        {n_heads_, head_size_, n_emb},
        tensor_name("v_weight"),
        dtype_,
        true);
    register_parameter("v_weight", w_v_);

    // w_o: (n_emb, n_heads, head_size)
    w_o_ = graph_->tensor(
        {n_emb, n_heads_, head_size_},
        tensor_name("o_weight"),
        dtype_,
        true);
    register_parameter("o_weight", w_o_);

    // Output projection bias
    out_bias_ = graph_->tensor(
        {n_emb},
        tensor_name("o_bias"),
        dtype_,
        true);
    register_parameter("o_bias", out_bias_);
}

graph::NNGraph::TensorNode* GptneoAttention::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* mask)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "GptneoAttention::forward: input tensor must be non-null");
    }

    // Q = gemm(w_q, x): (n_heads, head_size, n_emb) x (n_emb, seq, batch)
    graph::NNGraph::TensorNode* q_proj = graph::gemm(
        w_q_, x, tensor_name("q_proj"),
        1.0, false, false, 1, 0);
    graph::NNGraph::TensorNode* q =
        graph::transpose(q_proj, tensor_name("q"), 1);

    // K, V
    graph::NNGraph::TensorNode* k_proj = graph::gemm(
        w_k_, x, tensor_name("k_proj"),
        1.0, false, false, 1, 0);
    graph::NNGraph::TensorNode* k =
        graph::transpose(k_proj, tensor_name("k"), 1);

    graph::NNGraph::TensorNode* v_proj = graph::gemm(
        w_v_, x, tensor_name("v_proj"),
        1.0, false, false, 1, 0);
    graph::NNGraph::TensorNode* v =
        graph::transpose(v_proj, tensor_name("v"), 1);

    // SDPA (scale 1/sqrt(head_size) applied internally)
    graph::NNGraph::TensorNode* attn_out = graph::sdpa_eager(
        q, k, v,
        tensor_name("sdpa_out"),
        mask,
        2,  // batch_ndim
        0);

    // Transpose to (n_heads, head_size, seq, batch) for output projection
    graph::NNGraph::TensorNode* attn_t =
        graph::transpose(attn_out, tensor_name("attn_t"), 3);

    // Output projection: gemm(w_o, attn_t)
    graph::NNGraph::TensorNode* out = graph::gemm(
        w_o_, attn_t, tensor_name("out_proj"),
        1.0, false, false, 2, 0);

    // Add bias (fiber shape (n_emb,), tensor (n_emb, seq, batch), axis=0, batch_ndim=2)
    return graph::add_fiber(1.0, out_bias_, 1.0, out, tensor_name("attn_out"), 0, 2);
}

std::string GptneoAttention::repr() const
{
    return "GptneoAttention(hidden=" + std::to_string(config_.hidden_size) +
           ", n_heads=" + std::to_string(n_heads_) +
           ", head_size=" + std::to_string(head_size_) + ")";
}

} // namespace nntile::model::gptneo
