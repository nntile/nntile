#include <nntile/common.hh>
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

#include "nntile/nn/ops/concat.hh"
#include "nntile/nn/ops/gemm.hh"
#include "nntile/nn/ops/rope.hh"
#include "nntile/nn/ops/scale_slice.hh"
#include "nntile/nn/ops/sdpa_eager.hh"
#include "nntile/nn/ops/transpose.hh"
#include "nntile/tensor/ops/copy_intersection.hh"

#include <cmath>
#include <stdexcept>

namespace nntile::model::llama
{

LlamaAttention::LlamaAttention(NNGraph *graph,
    const std::string &name,
    const LlamaConfig &config,
    DataType dtype) :
    module::Module(graph, name),
    config_(config),
    dtype_(dtype),
    head_size_(config.head_dim),
    n_heads_(config.num_attention_heads),
    n_head_kv_(config.num_key_value_heads),
    kv_group_size_(config.num_attention_heads / config.num_key_value_heads),
    use_gqa_(config.num_key_value_heads < config.num_attention_heads)
{
    Index n_emb = config.hidden_size;

    if (use_gqa_)
    {
        w_q_ = graph_->tensor(
            {n_emb, head_size_, n_head_kv_, kv_group_size_}, dtype_, true);
        w_q_->set_name(tensor_name("q_weight"));
    }
    else
    {
        w_q_ = graph_->tensor({n_emb, head_size_, n_heads_}, dtype_, true);
        w_q_->set_name(tensor_name("q_weight"));
    }
    register_parameter("q_weight", w_q_);

    w_k_ = graph_->tensor({n_emb, head_size_, n_head_kv_}, dtype_, true);
    w_k_->set_name(tensor_name("k_weight"));
    register_parameter("k_weight", w_k_);

    w_v_ = graph_->tensor({n_emb, head_size_, n_head_kv_}, dtype_, true);
    w_v_->set_name(tensor_name("v_weight"));
    register_parameter("v_weight", w_v_);

    if (use_gqa_)
    {
        w_o_ = graph_->tensor(
            {head_size_, n_head_kv_, kv_group_size_, n_emb}, dtype_, true);
        w_o_->set_name(tensor_name("o_weight"));
    }
    else
    {
        w_o_ = graph_->tensor({head_size_, n_heads_, n_emb}, dtype_, true);
        w_o_->set_name(tensor_name("o_weight"));
    }
    register_parameter("o_weight", w_o_);
}

NNGraph::TensorNode *LlamaAttention::forward(
    NNGraph::TensorNode *x,
    NNGraph::TensorNode *sin,
    NNGraph::TensorNode *cos,
    NNGraph::TensorNode *mask,
    NNGraph::TensorNode *k_cache,
    NNGraph::TensorNode *v_cache,
    Index cache_len)
{
    if (x == nullptr)
    {
        throw std::invalid_argument(
            "LlamaAttention::forward: input tensor must be non-null");
    }

    const auto &x_shape = x->shape();
    Index n_batch = x_shape[0];
    Index n_seq = x_shape[1];

    NNGraph::TensorNode *q_proj;
    NNGraph::TensorNode *q;
    if (use_gqa_)
    {
        q_proj = gemm(w_q_, x, 1.0, false, false, 1, 0);
        q_proj->set_name(tensor_name("q_proj"));
        q = transpose(q_proj, 2);
        q->set_name(tensor_name("q"));
    }
    else
    {
        q_proj = gemm(w_q_, x, 1.0, false, false, 1, 0);
        q_proj->set_name(tensor_name("q_proj"));
        q = transpose(q_proj, 1);
        q->set_name(tensor_name("q"));
    }

    NNGraph::TensorNode *k_proj =
        gemm(w_k_, x, 1.0, false, false, 1, 0);
    k_proj->set_name(tensor_name("k_proj"));
    NNGraph::TensorNode *k = transpose(k_proj, 1);
    k->set_name(tensor_name("k"));

    NNGraph::TensorNode *v_proj =
        gemm(w_v_, x, 1.0, false, false, 1, 0);
    v_proj->set_name(tensor_name("v_proj"));
    NNGraph::TensorNode *v = transpose(v_proj, 1);
    v->set_name(tensor_name("v"));

    NNGraph::TensorNode *q_rope = q;
    NNGraph::TensorNode *k_rope = k;
    if (sin != nullptr && cos != nullptr)
    {
        q_rope = rope(sin, cos, q);
        q_rope->set_name(tensor_name("q_rope"));
        k_rope = rope(sin, cos, k);
        k_rope->set_name(tensor_name("k_rope"));
    }

    NNGraph::TensorNode *k_for_sdpa = k_rope;
    NNGraph::TensorNode *v_for_sdpa = v;
    if (k_cache != nullptr && v_cache != nullptr)
    {
        if (cache_len > 0)
        {
            NNGraph::TensorNode *k_cache_slice = graph_->tensor(
                {n_head_kv_, n_batch, cache_len, head_size_}, dtype_, false);
            k_cache_slice->set_name(tensor_name("k_cache_slice"));
            NNGraph::TensorNode *v_cache_slice = graph_->tensor(
                {n_head_kv_, n_batch, cache_len, head_size_}, dtype_, false);
            v_cache_slice->set_name(tensor_name("v_cache_slice"));
            tensor::copy_intersection(k_cache->data(),
                {0, 0, 0, 0},
                k_cache_slice->data(),
                {0, 0, 0, 0});
            tensor::copy_intersection(v_cache->data(),
                {0, 0, 0, 0},
                v_cache_slice->data(),
                {0, 0, 0, 0});
            k_for_sdpa = concat(k_cache_slice, k_rope, 2);
            k_for_sdpa->set_name(tensor_name("k_full"));
            v_for_sdpa = concat(v_cache_slice, v, 2);
            v_for_sdpa->set_name(tensor_name("v_full"));
        }
        tensor::copy_intersection(k_rope->data(),
            {0, 0, 0, 0},
            k_cache->data(),
            {0, cache_len, 0, 0});
        tensor::copy_intersection(
            v->data(), {0, 0, 0, 0}, v_cache->data(), {0, cache_len, 0, 0});
    }

    NNGraph::TensorNode *k_rep = k_for_sdpa;
    NNGraph::TensorNode *v_rep = v_for_sdpa;
    if (use_gqa_)
    {
        k_rep = scale_slice(1.0, k_for_sdpa, 1, kv_group_size_);
        k_rep->set_name(tensor_name("k_rep"));

        v_rep = scale_slice(1.0, v_for_sdpa, 1, kv_group_size_);
        v_rep->set_name(tensor_name("v_rep"));
    }

    Index batch_ndim = use_gqa_ ? 3 : 2;
    NNGraph::TensorNode *attn_out =
        sdpa_eager(q_rope, k_rep, v_rep, mask, batch_ndim, 0);
    attn_out->set_name(tensor_name("sdpa_out"));

    NNGraph::TensorNode *attn_t = transpose(attn_out, 3);
    attn_t->set_name(tensor_name("attn_t"));

    Index out_ndim = use_gqa_ ? 3 : 2;
    NNGraph::TensorNode *out =
        gemm(w_o_, attn_t, 1.0, false, false, out_ndim, 0);
    out->set_name(tensor_name("out_proj"));
    return out;
}

std::string LlamaAttention::repr() const
{
    return "LlamaAttention(hidden=" + std::to_string(config_.hidden_size) +
           ", n_heads=" + std::to_string(n_heads_) +
           ", head_size=" + std::to_string(head_size_) + ")";
}

} // namespace nntile::model::llama
