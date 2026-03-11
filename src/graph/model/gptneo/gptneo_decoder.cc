/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gptneo/gptneo_decoder.cc
 * GPT-Neo decoder block implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gptneo/gptneo_decoder.hh"
#include "nntile/graph/nn/add.hh"

#include <stdexcept>

namespace nntile::model::gptneo
{

GptneoDecoder::GptneoDecoder(graph::NNGraph* graph,
                             const std::string& name,
                             const GptneoConfig& config,
                             graph::DataType dtype)
    : graph::module::Module(graph, name)
    , input_norm_(graph, name + "_input_norm",
                  config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
    , attention_(graph, name + "_self_attn", config, dtype)
    , post_attn_norm_(graph, name + "_post_attn_norm",
                     config.hidden_size, 0, config.layer_norm_eps, 0, dtype)
    , mlp_(graph, name + "_mlp", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("input_norm", &input_norm_);
    register_module("attention", &attention_);
    register_module("post_attn_norm", &post_attn_norm_);
    register_module("mlp", &mlp_);
}

graph::NNGraph::TensorNode* GptneoDecoder::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* mask)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "GptneoDecoder::forward: input tensor must be non-null");
    }

    graph::NNGraph::TensorNode* x_norm = input_norm_.forward(x);
    graph::NNGraph::TensorNode* attn_out =
        attention_.forward(x_norm, mask);

    graph::NNGraph::TensorNode* post_attn =
        graph::add(1.0, x, 1.0, attn_out, tensor_name("post_attn"));

    graph::NNGraph::TensorNode* mlp_in = post_attn_norm_.forward(post_attn);
    graph::NNGraph::TensorNode* mlp_out = mlp_.forward(mlp_in);

    return graph::add(1.0, post_attn, 1.0, mlp_out, tensor_name("decoder_out"));
}

std::string GptneoDecoder::repr() const
{
    return "GptneoDecoder(hidden=" + std::to_string(config_.hidden_size) + ")";
}

} // namespace nntile::model::gptneo
