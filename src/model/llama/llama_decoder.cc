/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/model/llama/llama_decoder.cc
 * LlamaDecoder implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/llama/llama_decoder.hh"
#include "nntile/graph/nn/add.hh"

#include <stdexcept>

namespace nntile::model::llama
{

LlamaDecoder::LlamaDecoder(graph::NNGraph* graph,
                           const std::string& name,
                           const LlamaConfig& config,
                           graph::DataType dtype)
    : module::Module(graph, name)
    , input_norm_(graph, name + "_input_norm",
                  config.hidden_size, 2, config.rms_norm_eps, 0, dtype)  // axis=2 for (seq,batch,hidden)
    , attention_(graph, name + "_self_attn", config, dtype)
    , post_attn_norm_(graph, name + "_post_attn_norm",
                     config.hidden_size, 2, config.rms_norm_eps, 0, dtype)  // axis=2
    , mlp_(graph, name + "_mlp", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("input_norm", &input_norm_);
    register_module("attention", &attention_);
    register_module("post_attn_norm", &post_attn_norm_);
    register_module("mlp", &mlp_);
}

graph::NNGraph::TensorNode* LlamaDecoder::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* sin,
    graph::NNGraph::TensorNode* cos,
    graph::NNGraph::TensorNode* mask)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "LlamaDecoder::forward: input tensor must be non-null");
    }

    // input_norm -> attention
    graph::NNGraph::TensorNode* x_norm = input_norm_.forward(x);
    graph::NNGraph::TensorNode* attn_out =
        attention_.forward(x_norm, sin, cos, mask);

    // residual: x + attn_out
    graph::NNGraph::TensorNode* post_attn =
        graph::add(1.0, x, 1.0, attn_out, tensor_name("post_attn"));

    // post_attn_norm -> mlp
    graph::NNGraph::TensorNode* mlp_in = post_attn_norm_.forward(post_attn);
    graph::NNGraph::TensorNode* mlp_out = mlp_.forward(mlp_in);

    // residual: post_attn + mlp_out
    return graph::add(1.0, post_attn, 1.0, mlp_out, tensor_name("decoder_out"));
}

std::string LlamaDecoder::repr() const
{
    return "LlamaDecoder(hidden=" + std::to_string(config_.hidden_size) + ")";
}

} // namespace nntile::model::llama
