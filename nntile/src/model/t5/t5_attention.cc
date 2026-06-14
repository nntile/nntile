#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/t5/t5_attention.cc
 * T5Attention implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/t5/t5_attention.hh"
#include "nntile/nn/ops/gemm.hh"
#include "nntile/nn/ops/sdpa_eager.hh"
#include "nntile/nn/ops/transpose.hh"

#include <memory>
#include <stdexcept>

namespace nntile::model::t5
{

T5Attention::T5Attention(NNGraph* graph,
                         const std::string& name,
                         const T5Config& config,
                         bool is_cross_attention,
                         DataType dtype)
    : module::Module(graph, name)
    , config_(config)
    , dtype_(dtype)
    , is_cross_attention_(is_cross_attention)
    , head_size_(config.d_kv)
    , n_heads_(config.num_heads)
{
    Index d_model = config.d_model;

    w_q_ = graph_->tensor({d_model, head_size_, n_heads_}, dtype_, true);
    w_q_->set_name(tensor_name("q_weight"));
    register_parameter("q_weight", w_q_);

    w_k_ = graph_->tensor({d_model, head_size_, n_heads_}, dtype_, true);
    w_k_->set_name(tensor_name("k_weight"));
    register_parameter("k_weight", w_k_);

    w_v_ = graph_->tensor({d_model, head_size_, n_heads_}, dtype_, true);
    w_v_->set_name(tensor_name("v_weight"));
    register_parameter("v_weight", w_v_);

    w_o_ = graph_->tensor({head_size_, n_heads_, d_model}, dtype_, true);
    w_o_->set_name(tensor_name("o_weight"));
    register_parameter("o_weight", w_o_);
}

NNGraph::TensorNode* T5Attention::forward(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* encoder_output,
    NNGraph::TensorNode* mask)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "T5Attention::forward: input tensor must be non-null");
    }

    NNGraph::TensorNode* k_src = is_cross_attention_ && encoder_output
        ? encoder_output
        : x;
    NNGraph::TensorNode* v_src = k_src;

    NNGraph::TensorNode* q_proj =
        gemm(w_q_, x, 1.0, false, true, 1, 0);
    q_proj->set_name(tensor_name("q_proj"));
    NNGraph::TensorNode* q = transpose(q_proj, 3);
    q->set_name(tensor_name("q"));

    NNGraph::TensorNode* k_proj =
        gemm(w_k_, k_src, 1.0, false, true, 1, 0);
    k_proj->set_name(tensor_name("k_proj"));
    NNGraph::TensorNode* k = transpose(k_proj, 3);
    k->set_name(tensor_name("k"));

    NNGraph::TensorNode* v_proj =
        gemm(w_v_, v_src, 1.0, false, true, 1, 0);
    v_proj->set_name(tensor_name("v_proj"));
    NNGraph::TensorNode* v = transpose(v_proj, 3);
    v->set_name(tensor_name("v"));

    constexpr Scalar t5_attn_scale = 1.0;
    NNGraph* nn_graph = x->graph();
    auto sdpa_op = std::make_shared<NNSdpaEagerOp>(
        q, k, v, t5_attn_scale, 2, 0, mask);
    NNGraph::TensorNode* attn_out = sdpa_op->forward();
    nn_graph->register_op(std::move(sdpa_op));
    attn_out->set_name(tensor_name("sdpa_out"));

    NNGraph::TensorNode* attn_t = transpose(attn_out, 1);
    attn_t->set_name(tensor_name("attn_t"));

    NNGraph::TensorNode* out =
        gemm(w_o_, attn_t, 1.0, false, true, 2, 0);
    out->set_name(tensor_name("out_proj"));
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
