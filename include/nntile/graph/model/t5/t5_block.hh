/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/t5/t5_block.hh
 * T5EncoderBlock and T5DecoderBlock.
 *
 * Encoder: layer_norm -> self_attn -> add -> layer_norm -> ff -> add
 * Decoder: layer_norm -> self_attn -> add -> layer_norm -> cross_attn -> add
 *          -> layer_norm -> ff -> add
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/model/t5/t5_attention.hh>
#include <nntile/graph/model/t5/t5_config.hh>
#include <nntile/graph/model/t5/t5_ff.hh>
#include <nntile/graph/module/module.hh>
#include <nntile/graph/module/rms_norm.hh>

namespace nntile::model::t5
{

//! T5EncoderBlock - self_attn + ff with residuals
//! Flow: layer_norm_0 -> self_attn -> add -> ff (ff has layer_norm inside)
class T5EncoderBlock : public graph::module::Module
{
private:
    graph::module::RMSNorm layer_norm_0_;
    T5Attention self_attn_;
    T5LayerFF ff_;

    T5Config config_;
    graph::DataType dtype_;

public:
    T5EncoderBlock(graph::NNGraph* graph,
                   const std::string& name,
                   const T5Config& config,
                   graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;
};

//! T5DecoderBlock - self_attn + cross_attn + ff with residuals
//! Flow: ln0->self_attn->add -> ln1->cross_attn->add -> ff (ff has ln inside)
class T5DecoderBlock : public graph::module::Module
{
private:
    graph::module::RMSNorm layer_norm_0_;
    T5Attention self_attn_;
    graph::module::RMSNorm layer_norm_1_;
    T5Attention cross_attn_;
    T5LayerFF ff_;

    T5Config config_;
    graph::DataType dtype_;

public:
    T5DecoderBlock(graph::NNGraph* graph,
                   const std::string& name,
                   const T5Config& config,
                   graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* encoder_hidden_states,
        graph::NNGraph::TensorNode* self_attn_mask = nullptr,
        graph::NNGraph::TensorNode* cross_attn_mask = nullptr);

    std::string repr() const override;
};

} // namespace nntile::model::t5
