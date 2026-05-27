#include <nntile/graph/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/model/bert/bert_self_output.cc
 * BertSelfOutput implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/bert/bert_self_output.hh"
#include "nntile/graph/nn/ops/add.hh"
#include "nntile/graph/nn/ops/add_fiber.hh"
#include "nntile/graph/nn/ops/gemm.hh"

#include <stdexcept>

namespace nntile::graph::model::bert
{

BertSelfOutput::BertSelfOutput(graph::NNGraph* graph,
                               const std::string& name,
                               const BertConfig& config,
                               graph::DataType dtype)
    : graph::module::Module(graph, name)
    , layer_norm_(graph, name + "_ln",
                  config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();

    Index n_emb = config.hidden_size;
    Index n_heads = config.num_attention_heads;
    Index head_size = config.head_dim();

    w_dense_ = graph_->tensor({n_emb, n_heads, head_size}, dtype_, true);
    w_dense_->set_name(tensor_name("dense.weight"));
    register_parameter("dense.weight", w_dense_);

    b_dense_ = graph_->tensor({n_emb}, dtype_, true);
    b_dense_->set_name(tensor_name("dense.bias"));
    register_parameter("dense.bias", b_dense_);

    register_module("ln", &layer_norm_);
}

graph::NNGraph::TensorNode* BertSelfOutput::forward(
    graph::NNGraph::TensorNode* attn_heads,
    graph::NNGraph::TensorNode* residual)
{
    if(attn_heads == nullptr || residual == nullptr)
    {
        throw std::invalid_argument(
            "BertSelfOutput::forward: attn_heads and residual must be non-null");
    }

    graph::NNGraph::TensorNode* dense_out =
        graph::gemm(w_dense_, attn_heads, 1.0, false, false, 2, 0);
    dense_out = graph::add_fiber(1.0, b_dense_, 1.0, dense_out, 0, 0);
    dense_out->set_name(tensor_name("dense_out"));

    graph::NNGraph::TensorNode* summed =
        graph::add(1.0, residual, 1.0, dense_out);
    return layer_norm_.forward(summed);
}

std::string BertSelfOutput::repr() const
{
    return "BertSelfOutput(hidden=" + std::to_string(config_.hidden_size) + ")";
}

} // namespace nntile::graph::model::bert
