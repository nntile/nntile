#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/gpt2/gpt2_attention.cc
 * GPT2Attention implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gpt2/gpt2_attention.hh"
#include "nntile/model/gpt2/gpt2_common.hh"
#include "nntile/nn/ops/add_fiber.hh"
#include "nntile/nn/ops/gemm.hh"
#include "nntile/nn/ops/sdpa_eager.hh"

#include <stdexcept>

namespace nntile::model::gpt2
{

Gpt2Attention::Gpt2Attention(NNGraph* graph,
                             const std::string& name,
                             const Gpt2Config& config,
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

    w_o_ = graph_->tensor({head_size_, n_heads_, n_emb}, dtype_, true);
    w_o_->set_name(tensor_name("o_weight"));
    register_parameter("o_weight", w_o_);

    q_bias_ = graph_->tensor({head_size_, n_heads_}, dtype_, true);
    q_bias_->set_name(tensor_name("q_bias"));
    register_parameter("q_bias", q_bias_);

    k_bias_ = graph_->tensor({head_size_, n_heads_}, dtype_, true);
    k_bias_->set_name(tensor_name("k_bias"));
    register_parameter("k_bias", k_bias_);

    v_bias_ = graph_->tensor({head_size_, n_heads_}, dtype_, true);
    v_bias_->set_name(tensor_name("v_bias"));
    register_parameter("v_bias", v_bias_);

    o_bias_ = graph_->tensor({n_emb}, dtype_, true);
    o_bias_->set_name(tensor_name("o_bias"));
    register_parameter("o_bias", o_bias_);
}

NNGraph::TensorNode* Gpt2Attention::forward(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* mask,
    bool causal)
{
    throw_if_causal_flag_set(causal, "Gpt2Attention");

    if(x == nullptr)
    {
        throw std::invalid_argument(
            "Gpt2Attention::forward: input tensor must be non-null");
    }

    NNGraph::TensorNode* q_proj =
        gemm(x, w_q_, 1.0, false, false, 1, 0);
    q_proj->set_name(tensor_name("q_proj"));
    NNGraph::TensorNode* q = add_fiber(1.0, q_bias_, 1.0, q_proj, 2, 1);
    q->set_name(tensor_name("q"));

    NNGraph::TensorNode* k_proj =
        gemm(x, w_k_, 1.0, false, false, 1, 0);
    k_proj->set_name(tensor_name("k_proj"));
    NNGraph::TensorNode* k = add_fiber(1.0, k_bias_, 1.0, k_proj, 2, 1);
    k->set_name(tensor_name("k"));

    NNGraph::TensorNode* v_proj =
        gemm(x, w_v_, 1.0, false, false, 1, 0);
    v_proj->set_name(tensor_name("v_proj"));
    NNGraph::TensorNode* v = add_fiber(1.0, v_bias_, 1.0, v_proj, 2, 1);
    v->set_name(tensor_name("v"));

    NNGraph::TensorNode* attn_out =
        sdpa_eager(q, k, v, mask, 2, 0);
    attn_out->set_name(tensor_name("sdpa_out"));

    NNGraph::TensorNode* out =
        gemm(attn_out, w_o_, 1.0, false, false, 2, 0);
    const Index feature_axis = out->ndim() - 1;
    out = add_fiber(1.0, o_bias_, 1.0, out, feature_axis, 0);
    out->set_name(tensor_name("out_proj"));
    return out;
}

std::string Gpt2Attention::repr() const
{
    return "Gpt2Attention(hidden=" + std::to_string(config_.hidden_size) +
           ", n_heads=" + std::to_string(n_heads_) +
           ", head_size=" + std::to_string(head_size_) + ")";
}

} // namespace nntile::model::gpt2
