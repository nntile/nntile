/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/model/llama/llama_attention.cc
 * LlamaAttention implementation - gemm-based, mimics Python forward_async.
 *
 * @version 1.1.0
 * */

#include "nntile/model/llama/llama_attention.hh"
#include "nntile/graph/nn/gemm.hh"
#include "nntile/graph/nn/scale_slice.hh"
#include "nntile/graph/nn/rope.hh"
#include "nntile/graph/nn/sdpa_eager.hh"
#include "nntile/graph/nn/transpose.hh"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace nntile::model::llama
{

LlamaAttention::LlamaAttention(graph::NNGraph* graph,
                               const std::string& name,
                               const LlamaConfig& config,
                               graph::DataType dtype)
    : module::Module(graph, name)
    , config_(config)
    , dtype_(dtype)
    , head_size_(config.head_dim)
    , n_heads_(config.num_attention_heads)
    , n_head_kv_(config.num_key_value_heads)
    , kv_group_size_(config.num_attention_heads / config.num_key_value_heads)
    , use_gqa_(config.num_key_value_heads < config.num_attention_heads)
{
    Index n_emb = config.hidden_size;

    // Create weight tensors with 3D/4D shapes as in Python
    if(use_gqa_)
    {
        // w_q: (kv_group_size, n_head_kv, head_size, n_emb) - 4D
        w_q_ = graph_->tensor(
            {kv_group_size_, n_head_kv_, head_size_, n_emb},
            tensor_name("q_weight"),
            dtype_,
            true);
    }
    else
    {
        // w_q: (n_heads, head_size, n_emb) - 3D for non-GQA
        w_q_ = graph_->tensor(
            {n_heads_, head_size_, n_emb},
            tensor_name("q_weight"),
            dtype_,
            true);
    }
    register_parameter("q_weight", w_q_);

    // w_k, w_v: (n_head_kv, head_size, n_emb) - 3D
    w_k_ = graph_->tensor(
        {n_head_kv_, head_size_, n_emb},
        tensor_name("k_weight"),
        dtype_,
        true);
    register_parameter("k_weight", w_k_);

    w_v_ = graph_->tensor(
        {n_head_kv_, head_size_, n_emb},
        tensor_name("v_weight"),
        dtype_,
        true);
    register_parameter("v_weight", w_v_);

    if(use_gqa_)
    {
        // w_o: (n_emb, kv_group_size, n_head_kv, head_size) - 4D
        w_o_ = graph_->tensor(
            {n_emb, kv_group_size_, n_head_kv_, head_size_},
            tensor_name("o_weight"),
            dtype_,
            true);
    }
    else
    {
        // w_o: (n_emb, n_heads, head_size) - 3D for non-GQA
        w_o_ = graph_->tensor(
            {n_emb, n_heads_, head_size_},
            tensor_name("o_weight"),
            dtype_,
            true);
    }
    register_parameter("o_weight", w_o_);
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

    const auto& x_shape = x->shape();
    Index n_seq = x_shape[1];
    Index n_batch = x_shape[2];

    // Q = gemm(w_q, x), no transpose on x
    // x: (hidden, seq, batch), w_q: (..., n_emb) for contraction
    graph::NNGraph::TensorNode* q_proj;
    graph::NNGraph::TensorNode* q;
    if(use_gqa_)
    {
        // w_q (kv_group_size, n_head_kv, head_size, n_emb) x (n_emb, seq, batch)
        // gemm ndim=1 -> (kv_group_size, n_head_kv, head_size, seq, batch)
        q_proj = graph::gemm(
            w_q_, x, tensor_name("q_proj"),
            1.0, false, false, 1, 0);
        // transpose ndim=2: (d0,d1,d2,d3,d4) -> (d2,d3,d4,d0,d1)
        // (kv_group_size, n_head_kv, head_size, seq, batch) -> (head_size, seq, batch, kv_group_size, n_head_kv)
        q = graph::transpose(q_proj, tensor_name("q"), 2);
    }
    else
    {
        // w_q (n_heads, head_size, n_emb) x (n_emb, seq, batch)
        // gemm ndim=1 -> (n_heads, head_size, seq, batch)
        q_proj = graph::gemm(
            w_q_, x, tensor_name("q_proj"),
            1.0, false, false, 1, 0);
        // transpose ndim=1: (d0,d1,d2,d3) -> (d1,d2,d3,d0)
        // (n_heads, head_size, seq, batch) -> (head_size, seq, batch, n_heads)
        q = graph::transpose(q_proj, tensor_name("q"), 1);
    }

    // K = gemm(w_k, x), then transpose
    // w_k (n_head_kv, head_size, n_emb) x (n_emb, seq, batch)
    graph::NNGraph::TensorNode* k_proj = graph::gemm(
        w_k_, x, tensor_name("k_proj"),
        1.0, false, false, 1, 0);
    // transpose ndim=1: (n_head_kv, head_size, seq, batch) -> (head_size, seq, batch, n_head_kv)
    graph::NNGraph::TensorNode* k =
        graph::transpose(k_proj, tensor_name("k"), 1);

    // V = gemm(w_v, x), then transpose
    graph::NNGraph::TensorNode* v_proj = graph::gemm(
        w_v_, x, tensor_name("v_proj"),
        1.0, false, false, 1, 0);
    graph::NNGraph::TensorNode* v =
        graph::transpose(v_proj, tensor_name("v"), 1);

    // RoPE on Q and K (if sin/cos provided)
    graph::NNGraph::TensorNode* q_rope = q;
    graph::NNGraph::TensorNode* k_rope = k;
    if(sin != nullptr && cos != nullptr)
    {
        q_rope = graph::rope(sin, cos, q, tensor_name("q_rope"));
        k_rope = graph::rope(sin, cos, k, tensor_name("k_rope"));
    }

    // For GQA: repeat K and V to match Q's head count
    graph::NNGraph::TensorNode* k_rep = k_rope;
    graph::NNGraph::TensorNode* v_rep = v;
    if(use_gqa_)
    {
        // k_rope: (head_size, seq, batch, n_head_kv) - 4D
        // k_rep: (head_size, seq, batch, kv_group_size, n_head_kv) - 5D
        // scale_slice broadcasts k along axis 3
        k_rep = graph::scale_slice(
            1.0, k_rope, tensor_name("k_rep"), 3, kv_group_size_);

        v_rep = graph::scale_slice(
            1.0, v, tensor_name("v_rep"), 3, kv_group_size_);
    }

    // SDPA: q, k, v layout (head_size, seq, batch, ...)
    Index batch_ndim = use_gqa_ ? 3 : 2;
    graph::NNGraph::TensorNode* attn_out = graph::sdpa_eager(
        q_rope, k_rep, v_rep,
        tensor_name("sdpa_out"),
        mask,
        batch_ndim,
        0);

    // Transpose to (..., head_size) for output projection
    // attn_out: (head_size, seq, batch, ...) -> attn_t: (..., head_size, seq, batch)
    graph::NNGraph::TensorNode* attn_t =
        graph::transpose(attn_out, tensor_name("attn_t"), 3);

    // Output projection: gemm(w_o, attn_t)
    // w_o (n_emb, kv_group_size, n_head_kv, head_size) or (n_emb, n_heads, head_size)
    // attn_t (kv_group_size, n_head_kv, head_size, seq, batch) or (n_heads, head_size, seq, batch)
    // Contract last 2 dims of w_o with first 2 dims of attn_t for 3D w_o; last 3 with first 3 for 4D w_o
    Index out_ndim = use_gqa_ ? 3 : 2;
    graph::NNGraph::TensorNode* out = graph::gemm(
        w_o_, attn_t, tensor_name("out_proj"),
        1.0, false, false, out_ndim, 0);

    // Output is already (hidden, seq, batch)
    return out;
}

std::string LlamaAttention::repr() const
{
    return "LlamaAttention(hidden=" + std::to_string(config_.hidden_size) +
           ", n_heads=" + std::to_string(n_heads_) +
           ", head_size=" + std::to_string(head_size_) + ")";
}

namespace
{

// Convert HF (out, in) row-major to NNTile (n_heads, head_size, n_emb) col-major
void hf_linear_to_3d(const std::vector<std::uint8_t>& hf_data,
                    Index n_heads, Index head_size, Index n_emb,
                    std::vector<std::uint8_t>& out)
{
    const float* src = reinterpret_cast<const float*>(hf_data.data());
    float* dst = reinterpret_cast<float*>(out.data());
    for(Index h = 0; h < n_heads; ++h)
    {
        for(Index s = 0; s < head_size; ++s)
        {
            Index row = h * head_size + s;
            for(Index e = 0; e < n_emb; ++e)
            {
                dst[h + s * n_heads + e * n_heads * head_size] =
                    src[row * n_emb + e];
            }
        }
    }
}

// Convert NNTile (n_heads, head_size, n_emb) to HF (out, in) for q,k,v
void nntile_qkv_to_hf(const std::vector<std::uint8_t>& nntile_data,
                      Index n_heads, Index head_size, Index n_emb,
                      std::vector<std::uint8_t>& out)
{
    const float* src = reinterpret_cast<const float*>(nntile_data.data());
    float* dst = reinterpret_cast<float*>(out.data());
    for(Index h = 0; h < n_heads; ++h)
    {
        for(Index s = 0; s < head_size; ++s)
        {
            Index row = h * head_size + s;
            for(Index e = 0; e < n_emb; ++e)
            {
                dst[row * n_emb + e] =
                    src[h + s * n_heads + e * n_heads * head_size];
            }
        }
    }
}

// Convert NNTile (n_emb, n_heads, head_size) to HF (n_emb, n_emb) for o_proj
void nntile_o_to_hf(const std::vector<std::uint8_t>& nntile_data,
                    Index n_emb, Index n_heads, Index head_size,
                    std::vector<std::uint8_t>& out)
{
    const float* src = reinterpret_cast<const float*>(nntile_data.data());
    float* dst = reinterpret_cast<float*>(out.data());
    for(Index e = 0; e < n_emb; ++e)
    {
        for(Index h = 0; h < n_heads; ++h)
        {
            for(Index s = 0; s < head_size; ++s)
            {
                Index col = h * head_size + s;
                dst[e * n_emb + col] =
                    src[e + h * n_emb + s * n_emb * n_heads];
            }
        }
    }
}

// Convert HF (n_emb, n_emb) to NNTile (n_emb, n_heads, head_size) for o_proj
void hf_to_nntile_o(const std::vector<std::uint8_t>& hf_data,
                    Index n_emb, Index n_heads, Index head_size,
                    std::vector<std::uint8_t>& out)
{
    const float* src = reinterpret_cast<const float*>(hf_data.data());
    float* dst = reinterpret_cast<float*>(out.data());
    for(Index e = 0; e < n_emb; ++e)
    {
        for(Index h = 0; h < n_heads; ++h)
        {
            for(Index s = 0; s < head_size; ++s)
            {
                Index col = h * head_size + s;
                dst[e + h * n_emb + s * n_emb * n_heads] = src[e * n_emb + col];
            }
        }
    }
}

} // anonymous namespace

void LlamaAttention::import_hf(const io::SafeTensorsReader& reader,
                              const std::string& hf_prefix)
{
    Index n_emb = config_.hidden_size;
    const std::string prefix = hf_prefix.empty() ? "" : hf_prefix + ".";

    auto load_qkv = [&](const std::string& name, graph::NNGraph::TensorNode* param,
                       Index out_rows, Index out_cols)
    {
        const std::string key = prefix + name + ".weight";
        if(!reader.has_tensor(key))
        {
            throw std::runtime_error("LlamaAttention::import_hf: '" + key + "' not found");
        }
        const auto& info = reader.tensor_info(key);
        if(info.shape.size() != 2 || info.shape[0] != static_cast<std::int64_t>(out_rows) ||
           info.shape[1] != static_cast<std::int64_t>(out_cols))
        {
            throw std::runtime_error("LlamaAttention::import_hf: shape mismatch for " + key);
        }
        auto data = reader.read_tensor(key);
        const auto& param_shape = param->shape();
        std::vector<std::uint8_t> nntile_data(
            param_shape[0] * param_shape[1] * param_shape[2] * sizeof(float));
        hf_linear_to_3d(data, param_shape[0], param_shape[1], param_shape[2], nntile_data);
        param->data()->set_bind_hint(std::move(nntile_data));
        param->mark_input(true);
    };

    load_qkv("q_proj", w_q_, n_emb, n_emb);
    load_qkv("k_proj", w_k_, n_head_kv_ * head_size_, n_emb);
    load_qkv("v_proj", w_v_, n_head_kv_ * head_size_, n_emb);

    const std::string o_key = prefix + "o_proj.weight";
    if(!reader.has_tensor(o_key))
    {
        throw std::runtime_error("LlamaAttention::import_hf: '" + o_key + "' not found");
    }
    const auto& o_info = reader.tensor_info(o_key);
    if(o_info.shape.size() != 2 || o_info.shape[0] != static_cast<std::int64_t>(n_emb) ||
       o_info.shape[1] != static_cast<std::int64_t>(n_emb))
    {
        throw std::runtime_error("LlamaAttention::import_hf: shape mismatch for " + o_key);
    }
    auto o_data = reader.read_tensor(o_key);
    std::vector<std::uint8_t> o_nntile(n_emb * n_heads_ * head_size_ * sizeof(float));
    hf_to_nntile_o(o_data, n_emb, n_heads_, head_size_, o_nntile);
    w_o_->data()->set_bind_hint(std::move(o_nntile));
    w_o_->mark_input(true);
}

void LlamaAttention::export_hf(io::SafeTensorsWriter& writer,
                               const std::string& hf_prefix) const
{
    Index n_emb = config_.hidden_size;
    const std::string prefix = hf_prefix.empty() ? "" : hf_prefix + ".";

    auto export_qkv = [&](const std::string& name,
                          graph::NNGraph::TensorNode* param,
                          Index out_rows, Index out_cols)
    {
        const auto* hint = param->data()->get_bind_hint();
        if(hint == nullptr)
        {
            throw std::runtime_error("LlamaAttention::export_hf: " + name + " has no data");
        }
        const auto& shape = param->shape();
        std::vector<std::uint8_t> hf_data(out_rows * out_cols * sizeof(float));
        nntile_qkv_to_hf(*hint, shape[0], shape[1], shape[2], hf_data);
        writer.add_tensor(prefix + name + ".weight", dtype_,
                         {static_cast<std::int64_t>(out_rows),
                          static_cast<std::int64_t>(out_cols)},
                         std::move(hf_data));
    };

    export_qkv("q_proj", w_q_, n_emb, n_emb);
    export_qkv("k_proj", w_k_, n_head_kv_ * head_size_, n_emb);
    export_qkv("v_proj", w_v_, n_head_kv_ * head_size_, n_emb);

    const auto* o_hint = w_o_->data()->get_bind_hint();
    if(o_hint == nullptr)
    {
        throw std::runtime_error("LlamaAttention::export_hf: o_proj has no data");
    }
    std::vector<std::uint8_t> o_hf(n_emb * n_emb * sizeof(float));
    nntile_o_to_hf(*o_hint, n_emb, n_heads_, head_size_, o_hf);
    writer.add_tensor(prefix + "o_proj.weight", dtype_,
                     {static_cast<std::int64_t>(n_emb), static_cast<std::int64_t>(n_emb)},
                     std::move(o_hf));
}

} // namespace nntile::model::llama
