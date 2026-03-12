/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gpt2/gpt2_mlp.hh
 * GPT2MLP module - standard MLP (Linear -> GELU -> Linear).
 *
 * GPT-2 uses: c_fc (hidden -> 4*hidden) -> GELU -> c_proj (4*hidden -> hidden)
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/model/gpt2/gpt2_config.hh>
#include <nntile/graph/module/activation.hh>
#include <nntile/graph/module/mlp.hh>

namespace nntile::model::gpt2
{

//! GPT2MLP - standard MLP using GELU activation
class Gpt2MLP : public graph::module::Mlp
{
public:
    //! Constructor
    Gpt2MLP(graph::NNGraph* graph,
            const std::string& name,
            const Gpt2Config& config,
            graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input);

    std::string repr() const override;
};

} // namespace nntile::model::gpt2
