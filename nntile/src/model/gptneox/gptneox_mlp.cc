/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/gptneox/gptneox_mlp.cc
 * GPT-NeoXMLP implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gptneox/gptneox_mlp.hh"

namespace nntile::model::gptneox
{

GptneoxMlp::GptneoxMlp(NNGraph* graph,
                       const std::string& name,
                       const GptneoxConfig& config,
                       DataType dtype)
    : module::Mlp(graph,
                  name,
                  config.hidden_size,
                  config.intermediate_size,
                  config.hidden_size,
                  module::ActivationType::GELU,
                  true,
                  dtype)
{
    config.validate();
}

std::string GptneoxMlp::repr() const
{
    return "GptneoxMlp(" + module::Mlp::repr() + ")";
}

} // namespace nntile::model::gptneox
