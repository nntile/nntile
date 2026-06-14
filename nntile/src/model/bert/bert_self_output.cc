#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/bert/bert_self_output.cc
 * BertSelfOutput implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/bert/bert_self_output.hh"
#include "nntile/nn/ops/add.hh"
#include "nntile/nn/ops/add_fiber.hh"
#include "nntile/nn/ops/gemm.hh"

#include <stdexcept>

namespace nntile::model::bert
{

BertSelfOutput::BertSelfOutput(NNGraph* graph,
                               const std::string& name,
                               const BertConfig& config,
                               DataType dtype)
    : module::Module(graph, name)
    , layer_norm_(graph, name + "_ln",
                  config.hidden_size, 2, config.layer_norm_eps, 0, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();

    Index n_emb = config.hidden_size;
    Index n_heads = config.num_attention_heads;
    Index head_size = config.head_dim();

    w_dense_ = graph_->tensor({head_size, n_heads, n_emb}, dtype_, true);
    w_dense_->set_name(tensor_name("dense.weight"));
    register_parameter("dense.weight", w_dense_);

    b_dense_ = graph_->tensor({n_emb}, dtype_, true);
    b_dense_->set_name(tensor_name("dense.bias"));
    register_parameter("dense.bias", b_dense_);

    register_module("ln", &layer_norm_);
}

NNGraph::TensorNode* BertSelfOutput::forward(
    NNGraph::TensorNode* attn_heads,
    NNGraph::TensorNode* residual)
{
    if(attn_heads == nullptr || residual == nullptr)
    {
        throw std::invalid_argument(
            "BertSelfOutput::forward: attn_heads and residual must be non-null");
    }

    NNGraph::TensorNode* dense_out =
        gemm(w_dense_, attn_heads, 1.0, false, false, 2, 0);
    const Index feature_axis = dense_out->ndim() - 1;
    dense_out = add_fiber(1.0, b_dense_, 1.0, dense_out, feature_axis, 0);
    dense_out->set_name(tensor_name("dense_out"));

    NNGraph::TensorNode* summed =
        add(1.0, residual, 1.0, dense_out);
    return layer_norm_.forward(summed);
}

std::string BertSelfOutput::repr() const
{
    return "BertSelfOutput(hidden=" + std::to_string(config_.hidden_size) + ")";
}

} // namespace nntile::model::bert
