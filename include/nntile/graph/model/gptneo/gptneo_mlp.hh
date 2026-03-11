/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneo/gptneo_mlp.hh
 * GPT-Neo MLP module - up_proj -> GELU -> down_proj.
 *
 * Architecture: down_proj(GELU(up_proj(x)))
 * GPT-Neo uses gelutanh activation (GELU with tanh approximation).
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/model/gptneo/gptneo_config.hh>
#include <nntile/graph/module/activation.hh>
#include <nntile/graph/module/mlp.hh>

namespace nntile::model::gptneo
{

//! GPT-Neo MLP - up_proj -> GELU (gelutanh) -> down_proj
//! Architecture: down_proj(GELU(up_proj(x)))
class GptneoMLP : public graph::module::Mlp
{
public:
    //! Constructor
    //! @param graph Pointer to the neural network graph
    //! @param name Module name
    //! @param config GPT-Neo configuration
    //! @param dtype Data type
    GptneoMLP(graph::NNGraph* graph,
              const std::string& name,
              const GptneoConfig& config,
              graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input);

    //! Get string representation
    std::string repr() const override;
};

} // namespace nntile::model::gptneo
