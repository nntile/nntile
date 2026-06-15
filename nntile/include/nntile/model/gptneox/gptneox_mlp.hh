/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/gptneox/gptneox_mlp.hh
 * GPT-NeoX MLP module - up_proj -> GELU -> down_proj.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/gptneox/gptneox_config.hh>
#include <nntile/module/activation.hh>
#include <nntile/module/mlp.hh>

namespace nntile::model::gptneox
{

//! GPT-NeoXMLP - MLP with GELU and HF ``dense_*`` biases (like ``Gpt2MLP``).
class GptneoxMlp : public module::Mlp
{
public:
    GptneoxMlp(NNGraph* graph,
               const std::string& name,
               const GptneoxConfig& config,
               DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input);

    std::string repr() const override;
};

} // namespace nntile::model::gptneox
