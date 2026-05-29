/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/gptneo/gptneo_decoder.hh
 * GPT-Neo decoder block - ln_1 -> attention -> add -> ln_2 -> MLP -> add.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/gptneo/gptneo_attention.hh>
#include <nntile/model/gptneo/gptneo_config.hh>
#include <nntile/model/gptneo/gptneo_mlp.hh>
#include <nntile/module/module.hh>
#include <nntile/module/layer_norm.hh>

namespace nntile::model::gptneo
{

//! GPT-Neo decoder block: ln_1 -> attention -> residual -> ln_2 -> MLP -> residual
class GptneoDecoder : public module::Module
{
private:
    module::LayerNorm input_norm_;
    GptneoAttention attention_;
    module::LayerNorm post_attn_norm_;
    GptneoMLP mlp_;

    GptneoConfig config_;
    DataType dtype_;

public:
    //! Constructor
    GptneoDecoder(NNGraph* graph,
                  const std::string& name,
                  const GptneoConfig& config,
                  DataType dtype = DataType::FP32);

    //! Forward pass
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* x,
        NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    GptneoAttention& attention() { return attention_; }
    GptneoMLP& mlp() { return mlp_; }
};

} // namespace nntile::model::gptneo
