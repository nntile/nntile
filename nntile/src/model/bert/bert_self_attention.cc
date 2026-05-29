#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/model/bert/bert_self_attention.cc
 * BertSelfAttention implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/bert/bert_self_attention.hh"
#include "nntile/model/bert/bert_common.hh"
#include "nntile/nn_graph/ops/add_fiber.hh"
#include "nntile/nn_graph/ops/gemm.hh"
#include "nntile/nn_graph/ops/sdpa_eager.hh"
#include "nntile/nn_graph/ops/transpose.hh"

#include <stdexcept>

namespace nntile::model::bert
{

BertSelfAttention::BertSelfAttention(graph::NNGraph* graph,
                                     const std::string& name,
                                     const BertConfig& config,
                                     graph::DataType dtype)
    : graph::module::Module(graph, name)
    , config_(config)
    , dtype_(dtype)
    , head_size_(config.hidden_size / config.num_attention_heads)
    , n_heads_(config.num_attention_heads)
{
    config_.validate();

    Index n_emb = config.hidden_size;

    w_q_ = graph_->tensor({n_heads_, head_size_, n_emb}, dtype_, true);
    w_q_->set_name(tensor_name("q_weight"));
    register_parameter("q_weight", w_q_);

    w_k_ = graph_->tensor({n_heads_, head_size_, n_emb}, dtype_, true);
    w_k_->set_name(tensor_name("k_weight"));
    register_parameter("k_weight", w_k_);

    w_v_ = graph_->tensor({n_heads_, head_size_, n_emb}, dtype_, true);
    w_v_->set_name(tensor_name("v_weight"));
    register_parameter("v_weight", w_v_);

    q_bias_ = graph_->tensor({head_size_, n_heads_}, dtype_, true);
    q_bias_->set_name(tensor_name("q_bias"));
    register_parameter("q_bias", q_bias_);

    k_bias_ = graph_->tensor({head_size_, n_heads_}, dtype_, true);
    k_bias_->set_name(tensor_name("k_bias"));
    register_parameter("k_bias", k_bias_);

    v_bias_ = graph_->tensor({head_size_, n_heads_}, dtype_, true);
    v_bias_->set_name(tensor_name("v_bias"));
    register_parameter("v_bias", v_bias_);
}

graph::NNGraph::TensorNode* BertSelfAttention::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* mask,
    bool causal)
{
    throw_if_causal_flag_set(causal, "BertSelfAttention");

    if(x == nullptr)
    {
        throw std::invalid_argument(
            "BertSelfAttention::forward: input tensor must be non-null");
    }

    graph::NNGraph::TensorNode* q_proj =
        graph::gemm(w_q_, x, 1.0, false, false, 1, 0);
    q_proj->set_name(tensor_name("q_proj"));
    graph::NNGraph::TensorNode* q = graph::transpose(q_proj, 1);
    q = graph::add_fiber(1.0, q_bias_, 1.0, q, 0, 1);
    q->set_name(tensor_name("q"));

    graph::NNGraph::TensorNode* k_proj =
        graph::gemm(w_k_, x, 1.0, false, false, 1, 0);
    k_proj->set_name(tensor_name("k_proj"));
    graph::NNGraph::TensorNode* k = graph::transpose(k_proj, 1);
    k = graph::add_fiber(1.0, k_bias_, 1.0, k, 0, 1);
    k->set_name(tensor_name("k"));

    graph::NNGraph::TensorNode* v_proj =
        graph::gemm(w_v_, x, 1.0, false, false, 1, 0);
    v_proj->set_name(tensor_name("v_proj"));
    graph::NNGraph::TensorNode* v = graph::transpose(v_proj, 1);
    v = graph::add_fiber(1.0, v_bias_, 1.0, v, 0, 1);
    v->set_name(tensor_name("v"));

    graph::NNGraph::TensorNode* attn_out =
        graph::sdpa_eager(q, k, v, mask, 2, 0);
    attn_out->set_name(tensor_name("sdpa_out"));

    graph::NNGraph::TensorNode* attn_t = graph::transpose(attn_out, 3);
    attn_t->set_name(tensor_name("attn_heads"));
    return attn_t;
}

std::string BertSelfAttention::repr() const
{
    return "BertSelfAttention(hidden=" + std::to_string(config_.hidden_size) +
           ", n_heads=" + std::to_string(n_heads_) + ")";
}

} // namespace nntile::model::bert
