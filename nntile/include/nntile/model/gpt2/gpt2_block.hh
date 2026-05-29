/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/gpt2/gpt2_block.hh
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
#include <nntile/model/gpt2/gpt2_attention.hh>
#include <nntile/model/gpt2/gpt2_config.hh>
#include <nntile/model/gpt2/gpt2_mlp.hh>
#include <nntile/module/module.hh>
#include <nntile/module/layer_norm.hh>

namespace nntile::model::gpt2
{

//! GPT2Block - Pre-norm: ln_1 -> attention -> residual -> ln_2 -> mlp
class Gpt2Block : public module::Module
{
private:
    module::LayerNorm ln_1_;
    Gpt2Attention attention_;
    module::LayerNorm ln_2_;
    Gpt2MLP mlp_;

    Gpt2Config config_;
    DataType dtype_;

public:
    Gpt2Block(NNGraph* graph,
              const std::string& name,
              const Gpt2Config& config,
              DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* x,
        NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;
};

} // namespace nntile::model::gpt2
