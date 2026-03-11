/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gpt2/gpt2_block.hh
 * GPT2Block - one transformer block (attention + MLP with residuals).
 *
 * Pre-norm: ln_1 -> attention -> residual -> ln_2 -> mlp -> residual
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/model/gpt2/gpt2_attention.hh>
#include <nntile/graph/model/gpt2/gpt2_config.hh>
#include <nntile/graph/model/gpt2/gpt2_mlp.hh>
#include <nntile/graph/module/module.hh>
#include <nntile/graph/module/rms_norm.hh>

namespace nntile::model::gpt2
{

//! GPT2Block - Pre-norm: ln_1 -> attention -> residual -> ln_2 -> mlp -> residual
class Gpt2Block : public graph::module::Module
{
private:
    graph::module::RMSNorm ln_1_;
    Gpt2Attention attention_;
    graph::module::RMSNorm ln_2_;
    Gpt2MLP mlp_;

    Gpt2Config config_;
    graph::DataType dtype_;

public:
    //! Constructor
    Gpt2Block(graph::NNGraph* graph,
              const std::string& name,
              const Gpt2Config& config,
              graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;
};

} // namespace nntile::model::gpt2
