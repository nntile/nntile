/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/model/llama/llama_mlp.cc
 * LlamaMLP implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/llama/llama_mlp.hh"

namespace nntile::model::llama
{

LlamaMLP::LlamaMLP(graph::NNGraph* graph,
                   const std::string& name,
                   const LlamaConfig& config,
                   graph::DataType dtype)
    : module::GatedMlp(graph, name,
                       config.hidden_size,
                       config.intermediate_size,
                       config.hidden_size,
                       module::ActivationType::SILU,
                       dtype)
{
}

graph::NNGraph::TensorNode* LlamaMLP::forward(
    graph::NNGraph::TensorNode* input)
{
    return module::GatedMlp::forward(input);
}

std::string LlamaMLP::repr() const
{
    return "LlamaMLP(" + module::GatedMlp::repr() + ")";
}

} // namespace nntile::model::llama
