/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gpt2/gpt2_block.cc
 * GPT2Block implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gpt2/gpt2_block.hh"
#include "nntile/graph/nn/add.hh"

#include <stdexcept>

namespace nntile::model::gpt2
{

Gpt2Block::Gpt2Block(graph::NNGraph* graph,
                    const std::string& name,
                    const Gpt2Config& config,
                    graph::DataType dtype)
    : graph::module::Module(graph, name)
    , ln_1_(graph, name + "_ln_1",
            config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
    , attention_(graph, name + "_attn", config, dtype)
    , ln_2_(graph, name + "_ln_2",
            config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
    , mlp_(graph, name + "_mlp", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("ln_1", &ln_1_);
    register_module("attn", &attention_);
    register_module("ln_2", &ln_2_);
    register_module("mlp", &mlp_);
}

graph::NNGraph::TensorNode* Gpt2Block::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* mask)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "Gpt2Block::forward: input tensor must be non-null");
    }

    // ln_1 -> attention
    graph::NNGraph::TensorNode* x_norm = ln_1_.forward(x);
    graph::NNGraph::TensorNode* attn_out =
        attention_.forward(x_norm, mask);

    // residual: x + attn_out
    graph::NNGraph::TensorNode* post_attn =
        graph::add(1.0, x, 1.0, attn_out, tensor_name("post_attn"));

    // ln_2 -> mlp
    graph::NNGraph::TensorNode* mlp_in = ln_2_.forward(post_attn);
    graph::NNGraph::TensorNode* mlp_out = mlp_.forward(mlp_in);

    // residual: post_attn + mlp_out
    return graph::add(1.0, post_attn, 1.0, mlp_out, tensor_name("block_out"));
}

std::string Gpt2Block::repr() const
{
    return "Gpt2Block(hidden=" + std::to_string(config_.hidden_size) + ")";
}

} // namespace nntile::model::gpt2
