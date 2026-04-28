/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneox/gptneox_decoder.hh
 * GPT-NeoXDecoder - one transformer block with parallel residual.
 *
 * When use_parallel_residual: attention and MLP run in parallel from same
 * normalized input, then outputs are summed.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/model/gptneox/gptneox_attention.hh>
#include <nntile/graph/model/gptneox/gptneox_config.hh>
#include <nntile/graph/model/gptneox/gptneox_mlp.hh>
#include <nntile/graph/module/module.hh>
#include <nntile/graph/module/rms_norm.hh>

namespace nntile::model::gptneox
{

//! GPT-NeoXDecoder - input_norm -> [attention, post_attn_norm->mlp] with parallel residual
class GptneoxDecoder : public graph::module::Module
{
private:
    graph::module::RMSNorm input_norm_;
    GptneoxAttention attention_;
    graph::module::RMSNorm post_attn_norm_;
    GptneoxMlp mlp_;

    GptneoxConfig config_;
    graph::DataType dtype_;

public:
    //! Constructor
    GptneoxDecoder(graph::NNGraph* graph,
                   const std::string& name,
                   const GptneoxConfig& config,
                   graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* sin = nullptr,
        graph::NNGraph::TensorNode* cos = nullptr,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    GptneoxAttention& attention() { return attention_; }
    GptneoxMlp& mlp() { return mlp_; }
};

} // namespace nntile::model::gptneox
