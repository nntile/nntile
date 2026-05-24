/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gpt2/gpt2_attention.cc
 * GPT2Attention implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gpt2/gpt2_attention.hh"
#include "nntile/graph/nn/ops/gemm.hh"
#include "nntile/graph/nn/ops/sdpa_eager.hh"
#include "nntile/graph/nn/ops/transpose.hh"

#include <stdexcept>

namespace nntile::model::gpt2
{

Gpt2Attention::Gpt2Attention(graph::NNGraph* graph,
                             const std::string& name,
                             const Gpt2Config& config,
                             graph::DataType dtype)
    : graph::module::Module(graph, name)
    , config_(config)
    , dtype_(dtype)
    , head_size_(config.hidden_size / config.num_attention_heads)
    , n_heads_(config.num_attention_heads)
{
    Index n_emb = config.hidden_size;

    // w_q, w_k, w_v: (n_heads, head_size, n_emb) - 3D like LLaMA non-GQA
    w_q_ = graph_->tensor({n_heads_, head_size_, n_emb}, dtype_, true);

    w_q_->set_name(tensor_name("q_weight"));
    register_parameter("q_weight", w_q_);

    w_k_ = graph_->tensor({n_heads_, head_size_, n_emb}, dtype_, true);


    w_k_->set_name(tensor_name("k_weight"));
    register_parameter("k_weight", w_k_);

    w_v_ = graph_->tensor({n_heads_, head_size_, n_emb}, dtype_, true);


    w_v_->set_name(tensor_name("v_weight"));
    register_parameter("v_weight", w_v_);

    // w_o: (n_emb, n_heads, head_size) - 3D
    w_o_ = graph_->tensor({n_emb, n_heads_, head_size_}, dtype_, true);

    w_o_->set_name(tensor_name("o_weight"));
    register_parameter("o_weight", w_o_);
}

graph::NNGraph::TensorNode* Gpt2Attention::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* mask)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "Gpt2Attention::forward: input tensor must be non-null");
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

    graph::NNGraph::TensorNode* attn_out =
        graph::sdpa_eager(q, k, v, mask, 2, 0);
    attn_out->set_name(tensor_name("sdpa_out"));

    graph::NNGraph::TensorNode* attn_t = graph::transpose(attn_out, 3);
    attn_t->set_name(tensor_name("attn_t"));

    graph::NNGraph::TensorNode* out =
        graph::gemm(w_o_, attn_t, 1.0, false, false, 2, 0);
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
