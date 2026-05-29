/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/gpt2/gpt2_block.cc
 * GPT2Block implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gpt2/gpt2_block.hh"
#include "nntile/nn_graph/ops/add.hh"

#include <stdexcept>

namespace nntile::model::gpt2
{

Gpt2Block::Gpt2Block(NNGraph* graph,
                    const std::string& name,
                    const Gpt2Config& config,
                    DataType dtype)
    : module::Module(graph, name)
    , ln_1_(graph, name + "_ln_1",
            config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
    , attention_(graph, name + "_attn", config, dtype)
    , ln_2_(graph, name + "_ln_2",
            config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
    , mlp_(graph, name + "_mlp", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("ln_1", &ln_1_);
    register_module("attn", &attention_);
    register_module("ln_2", &ln_2_);
    register_module("mlp", &mlp_);
}

NNGraph::TensorNode* Gpt2Block::forward(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* mask,
    bool causal)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "Gpt2Block::forward: input tensor must be non-null");
    }

    NNGraph::TensorNode* x_norm = ln_1_.forward(x);
    NNGraph::TensorNode* attn_out =
        attention_.forward(x_norm, mask, causal);

    NNGraph::TensorNode* post_attn =
        add(1.0, x, 1.0, attn_out);

    NNGraph::TensorNode* mlp_in = ln_2_.forward(post_attn);
    NNGraph::TensorNode* mlp_out = mlp_.forward(mlp_in);

    return add(1.0, post_attn, 1.0, mlp_out);
}

std::string Gpt2Block::repr() const
{
    return "Gpt2Block(hidden=" + std::to_string(config_.hidden_size) + ")";
}

} // namespace nntile::model::gpt2
