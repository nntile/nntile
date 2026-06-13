/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/gptneo/gptneo_mlp.cc
 * GPT-Neo MLP implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gptneo/gptneo_mlp.hh"

namespace nntile::model::gptneo
{

GptneoMLP::GptneoMLP(NNGraph* graph,
                     const std::string& name,
                     const GptneoConfig& config,
                     DataType dtype)
    : module::Mlp(graph, name,
                         config.hidden_size,
                         config.intermediate_size,
                         config.hidden_size,
                         module::ActivationType::GELUTANH,
                         dtype)
{
}

NNGraph::TensorNode* GptneoMLP::forward(
    NNGraph::TensorNode* input)
{
    NNGraph::TensorNode* hidden = fc1().forward(input);
    hidden->set_name(tensor_name("fc1_out"));
    hidden = activation().forward(hidden);
    NNGraph::TensorNode* out = fc2().forward(hidden);
    out->set_name(tensor_name("mlp_out"));
    return out;
}

std::string GptneoMLP::repr() const
{
    return "GptneoMLP(" + module::Mlp::repr() + ")";
}

} // namespace nntile::model::gptneo
