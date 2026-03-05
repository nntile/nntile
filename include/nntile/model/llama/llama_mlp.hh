/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/llama/llama_mlp.hh
 * LlamaMLP module - gated MLP (SiLU(gate_proj(x)) * up_proj(x)) -> down_proj.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/llama/llama_config.hh>
#include <nntile/module/activation.hh>
#include <nntile/module/gated_mlp.hh>

namespace nntile::model::llama
{

//! LlamaMLP - Llama-style gated MLP using SiLU activation
//! Architecture: down_proj(SiLU(gate_proj(x)) * up_proj(x))
class LlamaMLP : public module::GatedMlp
{
public:
    //! Constructor
    //! @param graph Pointer to the neural network graph
    //! @param name Module name
    //! @param config Llama configuration
    LlamaMLP(graph::NNGraph* graph,
             const std::string& name,
             const LlamaConfig& config,
             graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input);

    //! Get string representation
    std::string repr() const override;
};

} // namespace nntile::model::llama
