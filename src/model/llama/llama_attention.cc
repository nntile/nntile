/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/model/llama/llama_attention.cc
 * LlamaAttention implementation (sdpa_eager-based).
 *
 * @version 1.1.0
 * */

#include "nntile/model/llama/llama_attention.hh"
#include "nntile/graph/nn/rope.hh"
#include "nntile/graph/nn/sdpa_eager.hh"
#include "nntile/graph/nn/transpose.hh"

#include <cmath>
#include <stdexcept>

namespace nntile::model::llama
{

LlamaAttention::LlamaAttention(graph::NNGraph* graph,
                               const std::string& name,
                               const LlamaConfig& config,
                               graph::DataType dtype)
    : module::Module(graph, name)
    , q_proj_(graph, name + "_q_proj",
              config.hidden_size,
              config.num_attention_heads * config.head_dim,
              config.attention_bias,
              dtype)
    , k_proj_(graph, name + "_k_proj",
              config.hidden_size,
              config.num_key_value_heads * config.head_dim,
              config.attention_bias,
              dtype)
    , v_proj_(graph, name + "_v_proj",
              config.hidden_size,
              config.num_key_value_heads * config.head_dim,
              config.attention_bias,
              dtype)
    , out_proj_(graph, name + "_o_proj",
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
                config.attention_bias,
                dtype)
    , config_(config)
    , dtype_(dtype)
    , head_size_(config.head_dim)
    , n_heads_(config.num_attention_heads)
    , n_head_kv_(config.num_key_value_heads)
    , kv_group_size_(config.num_attention_heads / config.num_key_value_heads)
{
    if(n_heads_ != n_head_kv_)
    {
        throw std::invalid_argument(
            "LlamaAttention: GQA (num_key_value_heads < num_attention_heads) "
            "requires repeat_kv which needs reshape - not yet supported");
    }
    register_module("q_proj", &q_proj_);
    register_module("k_proj", &k_proj_);
    register_module("v_proj", &v_proj_);
    register_module("out_proj", &out_proj_);
}

graph::NNGraph::TensorNode* LlamaAttention::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* sin,
    graph::NNGraph::TensorNode* cos,
    graph::NNGraph::TensorNode* mask)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "LlamaAttention::forward: input tensor must be non-null");
    }

    // Q, K, V projections: (seq, batch, hidden) -> (seq, batch, head_size)
    graph::NNGraph::TensorNode* q = q_proj_.forward(x);
    graph::NNGraph::TensorNode* k = k_proj_.forward(x);
    graph::NNGraph::TensorNode* v = v_proj_.forward(x);

    // Transpose to (head_size, seq, batch) for sdpa_eager (ndim=2 cyclic shift)
    graph::NNGraph::TensorNode* q_t =
        graph::transpose(q, tensor_name("q_t"), 2);
    graph::NNGraph::TensorNode* k_t =
        graph::transpose(k, tensor_name("k_t"), 2);
    graph::NNGraph::TensorNode* v_t =
        graph::transpose(v, tensor_name("v_t"), 2);

    // RoPE on Q and K (if sin/cos provided)
    graph::NNGraph::TensorNode* q_rope = q_t;
    graph::NNGraph::TensorNode* k_rope = k_t;
    if(sin != nullptr && cos != nullptr)
    {
        q_rope = graph::rope(sin, cos, q_t, tensor_name("q_rope"));
        k_rope = graph::rope(sin, cos, k_t, tensor_name("k_rope"));
    }

    // SDPA: batch_ndim=1 for (head_size, seq, batch)
    graph::NNGraph::TensorNode* attn_out = graph::sdpa_eager(
        q_rope, k_rope, v_t,
        tensor_name("attn_out"),
        mask,
        1,  // batch_ndim
        0   // redux
    );

    // Transpose back to (seq, batch, head_size) - ndim=1 for 3D
    graph::NNGraph::TensorNode* attn_t =
        graph::transpose(attn_out, tensor_name("attn_t"), 1);

    // Output projection
    return out_proj_.forward(attn_t);
}

std::string LlamaAttention::repr() const
{
    return "LlamaAttention(hidden=" + std::to_string(config_.hidden_size) +
           ", n_heads=" + std::to_string(n_heads_) +
           ", head_size=" + std::to_string(head_size_) + ")";
}

} // namespace nntile::model::llama
