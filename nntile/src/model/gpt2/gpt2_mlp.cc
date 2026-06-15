/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/gpt2/gpt2_mlp.cc
 * GPT2MLP implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gpt2/gpt2_mlp.hh"

namespace nntile::model::gpt2
{

Gpt2MLP::Gpt2MLP(NNGraph* graph,
                 const std::string& name,
                 const Gpt2Config& config,
                 DataType dtype)
    : module::Mlp(graph,
                  name,
                  config.hidden_size,
                  config.intermediate_size,
                  config.hidden_size,
                  module::ActivationType::GELUTANH,
                  true,
                  dtype)
{
    config.validate();
}

std::string Gpt2MLP::repr() const
{
    return "Gpt2MLP(" + module::Mlp::repr() + ")";
}

} // namespace nntile::model::gpt2
