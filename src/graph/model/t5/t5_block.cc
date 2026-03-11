/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/t5/t5_block.cc
 * T5EncoderBlock and T5DecoderBlock implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/t5/t5_block.hh"
#include "nntile/graph/nn/add.hh"

#include <stdexcept>

namespace nntile::model::t5
{

// ── T5EncoderBlock ─────────────────────────────────────────────────────────

T5EncoderBlock::T5EncoderBlock(graph::NNGraph* graph,
                               const std::string& name,
                               const T5Config& config,
                               graph::DataType dtype)
    : graph::module::Module(graph, name)
    , layer_norm_0_(graph, name + "_layer_norm_0",
                    config.d_model, 0, config.layer_norm_epsilon, 0, dtype)
    , self_attn_(graph, name + "_self_attn", config, false, dtype)
    , ff_(graph, name + "_ff", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("layer_norm_0", &layer_norm_0_);
    register_module("self_attn", &self_attn_);
    register_module("ff", &ff_);
}

graph::NNGraph::TensorNode* T5EncoderBlock::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* mask)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "T5EncoderBlock::forward: input tensor must be non-null");
    }

    // layer_norm_0 -> self_attn -> add
    graph::NNGraph::TensorNode* x_norm = layer_norm_0_.forward(x);
    graph::NNGraph::TensorNode* attn_out =
        self_attn_.forward(x_norm, nullptr, mask);
    graph::NNGraph::TensorNode* post_attn =
        graph::add(1.0, x, 1.0, attn_out, tensor_name("post_attn"));

    // layer_norm_1 -> ff (ff includes residual)
    return ff_.forward(post_attn);
}

std::string T5EncoderBlock::repr() const
{
    return "T5EncoderBlock(d_model=" + std::to_string(config_.d_model) + ")";
}

// ── T5DecoderBlock ─────────────────────────────────────────────────────────

T5DecoderBlock::T5DecoderBlock(graph::NNGraph* graph,
                               const std::string& name,
                               const T5Config& config,
                               graph::DataType dtype)
    : graph::module::Module(graph, name)
    , layer_norm_0_(graph, name + "_layer_norm_0",
                    config.d_model, 0, config.layer_norm_epsilon, 0, dtype)
    , self_attn_(graph, name + "_self_attn", config, false, dtype)
    , layer_norm_1_(graph, name + "_layer_norm_1",
                    config.d_model, 0, config.layer_norm_epsilon, 0, dtype)
    , cross_attn_(graph, name + "_cross_attn", config, true, dtype)
    , ff_(graph, name + "_ff", config, dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("layer_norm_0", &layer_norm_0_);
    register_module("self_attn", &self_attn_);
    register_module("layer_norm_1", &layer_norm_1_);
    register_module("cross_attn", &cross_attn_);
    register_module("ff", &ff_);
}

graph::NNGraph::TensorNode* T5DecoderBlock::forward(
    graph::NNGraph::TensorNode* x,
    graph::NNGraph::TensorNode* encoder_hidden_states,
    graph::NNGraph::TensorNode* self_attn_mask,
    graph::NNGraph::TensorNode* cross_attn_mask)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "T5DecoderBlock::forward: input tensor must be non-null");
    }
    if(encoder_hidden_states == nullptr)
    {
        throw std::invalid_argument(
            "T5DecoderBlock::forward: encoder_hidden_states must be non-null");
    }

    // Self-attention
    graph::NNGraph::TensorNode* x_norm = layer_norm_0_.forward(x);
    graph::NNGraph::TensorNode* self_attn_out =
        self_attn_.forward(x_norm, nullptr, self_attn_mask);
    graph::NNGraph::TensorNode* post_self =
        graph::add(1.0, x, 1.0, self_attn_out, tensor_name("post_self_attn"));

    // Cross-attention
    graph::NNGraph::TensorNode* x_norm1 = layer_norm_1_.forward(post_self);
    graph::NNGraph::TensorNode* cross_attn_out =
        cross_attn_.forward(x_norm1, encoder_hidden_states, cross_attn_mask);
    graph::NNGraph::TensorNode* post_cross =
        graph::add(1.0, post_self, 1.0, cross_attn_out, tensor_name("post_cross_attn"));

    // FF
    return ff_.forward(post_cross);
}

std::string T5DecoderBlock::repr() const
{
    return "T5DecoderBlock(d_model=" + std::to_string(config_.d_model) + ")";
}

} // namespace nntile::model::t5
