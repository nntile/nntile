#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/gptneox/gptneox_attention.cc
 * GPT-NeoXAttention implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gptneox/gptneox_attention.hh"
#include "nntile/nn_graph/ops/gemm.hh"
#include "nntile/nn_graph/ops/rope.hh"
#include "nntile/nn_graph/ops/sdpa_eager.hh"
#include "nntile/nn_graph/ops/transpose.hh"

#include <stdexcept>

namespace nntile::model::gptneox
{

GptneoxAttention::GptneoxAttention(graph::NNGraph* graph,
                                   const std::string& name,
                                   const GptneoxConfig& config,
                                   graph::DataType dtype)
    : graph::module::Module(graph, name)
    , config_(config)
    , dtype_(dtype)
    , head_size_(config.head_dim)
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

    w_o_ = graph_->tensor({n_emb, n_heads_, head_size_}, dtype_, true);
    w_o_->set_name(tensor_name("o_weight"));
    register_parameter("o_weight", w_o_);
}

graph::NNGraph::TensorNode* GptneoxAttention::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* sin,
    graph::NNGraph::TensorNode* cos,
    graph::NNGraph::TensorNode* mask)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "GptneoxAttention::forward: input tensor must be non-null");
    }

    graph::NNGraph::TensorNode* q_proj =
        graph::gemm(w_q_, x, 1.0, false, false, 1, 0);
    q_proj->set_name(tensor_name("q_proj"));
    graph::NNGraph::TensorNode* q = graph::transpose(q_proj, 1);
    q->set_name(tensor_name("q"));

    graph::NNGraph::TensorNode* k_proj =
        graph::gemm(w_k_, x, 1.0, false, false, 1, 0);
    k_proj->set_name(tensor_name("k_proj"));
    graph::NNGraph::TensorNode* k = graph::transpose(k_proj, 1);
    k->set_name(tensor_name("k"));

    graph::NNGraph::TensorNode* v_proj =
        graph::gemm(w_v_, x, 1.0, false, false, 1, 0);
    v_proj->set_name(tensor_name("v_proj"));
    graph::NNGraph::TensorNode* v = graph::transpose(v_proj, 1);
    v->set_name(tensor_name("v"));

    graph::NNGraph::TensorNode* q_rope = q;
    graph::NNGraph::TensorNode* k_rope = k;
    if(sin != nullptr && cos != nullptr)
    {
        q_rope = graph::rope(sin, cos, q);
        q_rope->set_name(tensor_name("q_rope"));
        k_rope = graph::rope(sin, cos, k);
        k_rope->set_name(tensor_name("k_rope"));
    }

    graph::NNGraph::TensorNode* attn_out =
        graph::sdpa_eager(q_rope, k_rope, v, mask, 2, 0);
    attn_out->set_name(tensor_name("sdpa_out"));

    graph::NNGraph::TensorNode* attn_t = graph::transpose(attn_out, 3);
    attn_t->set_name(tensor_name("attn_t"));

    graph::NNGraph::TensorNode* out =
        graph::gemm(w_o_, attn_t, 1.0, false, false, 2, 0);
    out->set_name(tensor_name("out_proj"));
    return out;
}

std::string GptneoxAttention::repr() const
{
    return "GptneoxAttention(hidden=" + std::to_string(config_.hidden_size) +
           ", n_heads=" + std::to_string(n_heads_) +
           ", head_size=" + std::to_string(head_size_) + ")";
}

} // namespace nntile::model::gptneox
