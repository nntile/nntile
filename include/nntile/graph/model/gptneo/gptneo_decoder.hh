/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneo/gptneo_decoder.hh
 * GPT-Neo decoder block - ln_1 -> attention -> add -> ln_2 -> MLP -> add.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/model/gptneo/gptneo_attention.hh>
#include <nntile/graph/model/gptneo/gptneo_config.hh>
#include <nntile/graph/model/gptneo/gptneo_mlp.hh>
#include <nntile/graph/module/module.hh>
#include <nntile/graph/module/rms_norm.hh>

namespace nntile::model::gptneo
{

//! GPT-Neo decoder block: ln_1 -> attention -> residual -> ln_2 -> MLP -> residual
class GptneoDecoder : public graph::module::Module
{
private:
    graph::module::RMSNorm input_norm_;
    GptneoAttention attention_;
    graph::module::RMSNorm post_attn_norm_;
    GptneoMLP mlp_;

    GptneoConfig config_;
    graph::DataType dtype_;

public:
    //! Constructor
    GptneoDecoder(graph::NNGraph* graph,
                  const std::string& name,
                  const GptneoConfig& config,
                  graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    GptneoAttention& attention() { return attention_; }
    GptneoMLP& mlp() { return mlp_; }
};

} // namespace nntile::model::gptneo
