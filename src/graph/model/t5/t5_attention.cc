/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/t5/t5_attention.cc
 * T5Attention implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/t5/t5_attention.hh"
#include "nntile/graph/nn/gemm.hh"
#include "nntile/graph/nn/sdpa_eager.hh"
#include "nntile/graph/nn/transpose.hh"

#include <cmath>
#include <stdexcept>

namespace nntile::model::t5
{

T5Attention::T5Attention(graph::NNGraph* graph,
                         const std::string& name,
                         const T5Config& config,
                         bool is_cross_attention,
                         graph::DataType dtype)
    : graph::module::Module(graph, name)
    , config_(config)
    , dtype_(dtype)
    , is_cross_attention_(is_cross_attention)
    , head_size_(config.d_kv)
    , n_heads_(config.num_heads)
{
    Index d_model = config.d_model;

    // w_q: (n_heads, head_size, d_model)
    w_q_ = graph_->tensor(
        {n_heads_, head_size_, d_model},
        tensor_name("q_weight"),
        dtype_,
        true);
    register_parameter("q_weight", w_q_);

    // w_k, w_v: (n_heads, head_size, d_model) for self-attn
    // For cross-attn, K,V come from encoder: (n_heads, head_size, d_model) - same
    w_k_ = graph_->tensor(
        {n_heads_, head_size_, d_model},
        tensor_name("k_weight"),
        dtype_,
        true);
    register_parameter("k_weight", w_k_);

    w_v_ = graph_->tensor(
        {n_heads_, head_size_, d_model},
        tensor_name("v_weight"),
        dtype_,
        true);
    register_parameter("v_weight", w_v_);

    // w_o: (d_model, n_heads, head_size)
    w_o_ = graph_->tensor(
        {d_model, n_heads_, head_size_},
        tensor_name("o_weight"),
        dtype_,
        true);
    register_parameter("o_weight", w_o_);
}

graph::NNGraph::TensorNode* T5Attention::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* encoder_output,
    graph::NNGraph::TensorNode* mask)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "T5Attention::forward: input tensor must be non-null");
    }

    graph::NNGraph::TensorNode* k_src = is_cross_attention_ && encoder_output
        ? encoder_output
        : x;
    graph::NNGraph::TensorNode* v_src = k_src;

    // Q = gemm(w_q, x)
    graph::NNGraph::TensorNode* q_proj = graph::gemm(
        w_q_, x, tensor_name("q_proj"),
        1.0, false, false, 1, 0);
    graph::NNGraph::TensorNode* q =
        graph::transpose(q_proj, tensor_name("q"), 1);

    // K = gemm(w_k, k_src)
    graph::NNGraph::TensorNode* k_proj = graph::gemm(
        w_k_, k_src, tensor_name("k_proj"),
        1.0, false, false, 1, 0);
    graph::NNGraph::TensorNode* k =
        graph::transpose(k_proj, tensor_name("k"), 1);

    // V = gemm(w_v, v_src)
    graph::NNGraph::TensorNode* v_proj = graph::gemm(
        w_v_, v_src, tensor_name("v_proj"),
        1.0, false, false, 1, 0);
    graph::NNGraph::TensorNode* v =
        graph::transpose(v_proj, tensor_name("v"), 1);

    // SDPA with scale 1/sqrt(head_size)
    Scalar scale = 1.0 / std::sqrt(static_cast<double>(head_size_));
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

    return out;
}

std::string T5Attention::repr() const
{
    return "T5Attention(d_model=" + std::to_string(config_.d_model) +
           ", n_heads=" + std::to_string(n_heads_) +
           ", head_size=" + std::to_string(head_size_) +
           ", cross=" + (is_cross_attention_ ? "true" : "false") + ")";
}

} // namespace nntile::model::t5
