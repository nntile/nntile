/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/t5/t5_block.hh
 * T5EncoderBlock and T5DecoderBlock.
 *
 * Encoder: layer_norm -> self_attn -> add -> ff (internal ln + residual)
 * Decoder: ln0 -> self_attn -> add -> ln1 -> cross_attn -> add -> ff
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/t5/t5_attention.hh>
#include <nntile/model/t5/t5_config.hh>
#include <nntile/model/t5/t5_ff.hh>
#include <nntile/module/module.hh>
#include <nntile/module/rms_norm.hh>

namespace nntile::model::t5
{

//! T5EncoderBlock - self_attn + ff with residuals
//! Flow: layer_norm_0 -> self_attn -> add -> ff (ff has layer_norm inside)
class T5EncoderBlock : public module::Module
{
private:
    module::RMSNorm layer_norm_0_;
    T5Attention self_attn_;
    T5LayerFF ff_;

    T5Config config_;
    DataType dtype_;

public:
    T5EncoderBlock(NNGraph* graph,
                   const std::string& name,
                   const T5Config& config,
                   DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* x,
        NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;
};

//! T5DecoderBlock - self_attn + cross_attn + ff with residuals
//! Flow: ln0->self_attn->add -> ln1->cross_attn->add -> ff (ff has ln inside)
class T5DecoderBlock : public module::Module
{
private:
    module::RMSNorm layer_norm_0_;
    T5Attention self_attn_;
    module::RMSNorm layer_norm_1_;
    T5Attention cross_attn_;
    T5LayerFF ff_;

    T5Config config_;
    DataType dtype_;

public:
    T5DecoderBlock(NNGraph* graph,
                   const std::string& name,
                   const T5Config& config,
                   DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* x,
        NNGraph::TensorNode* encoder_hidden_states,
        NNGraph::TensorNode* self_attn_mask = nullptr,
        NNGraph::TensorNode* cross_attn_mask = nullptr);

    std::string repr() const override;
};

} // namespace nntile::model::t5
