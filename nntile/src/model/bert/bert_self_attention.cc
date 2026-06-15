#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/bert/bert_self_attention.cc
 * BertSelfAttention implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/bert/bert_self_attention.hh"
#include "nntile/model/bert/bert_common.hh"
#include "nntile/nn/ops/add_fiber.hh"
#include "nntile/nn/ops/gemm.hh"
#include "nntile/nn/ops/sdpa_eager.hh"
#include "nntile/nn/ops/transpose.hh"

#include <stdexcept>

namespace nntile::model::bert
{

BertSelfAttention::BertSelfAttention(NNGraph* graph,
                                     const std::string& name,
                                     const BertConfig& config,
                                     DataType dtype)
    : module::Module(graph, name)
    , config_(config)
    , dtype_(dtype)
    , head_size_(config.hidden_size / config.num_attention_heads)
    , n_heads_(config.num_attention_heads)
{
    config_.validate();

    Index n_emb = config.hidden_size;

    w_q_ = graph_->tensor({n_emb, head_size_, n_heads_}, dtype_, true);
    w_q_->set_name(tensor_name("q_weight"));
    register_parameter("q_weight", w_q_);

    w_k_ = graph_->tensor({n_emb, head_size_, n_heads_}, dtype_, true);
    w_k_->set_name(tensor_name("k_weight"));
    register_parameter("k_weight", w_k_);

    w_v_ = graph_->tensor({n_emb, head_size_, n_heads_}, dtype_, true);
    w_v_->set_name(tensor_name("v_weight"));
    register_parameter("v_weight", w_v_);

    q_bias_ = graph_->tensor({n_heads_, head_size_}, dtype_, true);
    q_bias_->set_name(tensor_name("q_bias"));
    register_parameter("q_bias", q_bias_);

    k_bias_ = graph_->tensor({n_heads_, head_size_}, dtype_, true);
    k_bias_->set_name(tensor_name("k_bias"));
    register_parameter("k_bias", k_bias_);

    v_bias_ = graph_->tensor({n_heads_, head_size_}, dtype_, true);
    v_bias_->set_name(tensor_name("v_bias"));
    register_parameter("v_bias", v_bias_);
}

NNGraph::TensorNode* BertSelfAttention::forward(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* mask,
    bool causal,
    NNGraph::TensorNode* w_dense,
    NNGraph::TensorNode* b_dense)
{
    throw_if_causal_flag_set(causal, "BertSelfAttention");

    if(x == nullptr)
    {
        throw std::invalid_argument(
            "BertSelfAttention::forward: input tensor must be non-null");
    }

    NNGraph::TensorNode* q_proj =
        gemm(x, w_q_, 1.0, false, false, 1, 0);
    q_proj->set_name(tensor_name("q_proj"));
    NNGraph::TensorNode* q = transpose(q_proj, 1);
    q = add_fiber(1.0, q_bias_, 1.0, q, 3, 1);
    q->set_name(tensor_name("q"));

    NNGraph::TensorNode* k_proj =
        gemm(x, w_k_, 1.0, false, false, 1, 0);
    k_proj->set_name(tensor_name("k_proj"));
    NNGraph::TensorNode* k = transpose(k_proj, 1);
    k = add_fiber(1.0, k_bias_, 1.0, k, 3, 1);
    k->set_name(tensor_name("k"));

    NNGraph::TensorNode* v_proj =
        gemm(x, w_v_, 1.0, false, false, 1, 0);
    v_proj->set_name(tensor_name("v_proj"));
    NNGraph::TensorNode* v = transpose(v_proj, 1);
    v = add_fiber(1.0, v_bias_, 1.0, v, 3, 1);
    v->set_name(tensor_name("v"));

    NNGraph::TensorNode* attn_out =
        sdpa_eager(q, k, v, mask, 2, 0);
    attn_out->set_name(tensor_name("sdpa_out"));

    NNGraph::TensorNode* attn_t = transpose(attn_out, 3);
    attn_t->set_name(tensor_name("attn_t"));

    if(w_dense == nullptr || b_dense == nullptr)
    {
        return attn_t;
    }

    NNGraph::TensorNode* out =
        gemm(attn_t, w_dense, 1.0, false, false, 2, 0);
    out = add_fiber(1.0, b_dense, 1.0, out, out->ndim() - 1, 0);
    out->set_name(tensor_name("dense_out"));
    return out;
}

std::string BertSelfAttention::repr() const
{
    return "BertSelfAttention(hidden=" + std::to_string(config_.hidden_size) +
           ", n_heads=" + std::to_string(n_heads_) + ")";
}

} // namespace nntile::model::bert
