/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gptneo/gptneo_mlp.cc
 * GPT-Neo MLP implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gptneo/gptneo_mlp.hh"
#include "nntile/graph/nn/ops/gemm.hh"

namespace nntile::model::gptneo
{

GptneoMLP::GptneoMLP(graph::NNGraph* graph,
                     const std::string& name,
                     const GptneoConfig& config,
                     graph::DataType dtype)
    : graph::module::Mlp(graph, name,
                         config.hidden_size,
                         config.intermediate_size,
                         config.hidden_size,
                         graph::module::ActivationType::GELUTANH,
                         dtype)
{
}

graph::NNGraph::TensorNode* GptneoMLP::forward(
    graph::NNGraph::TensorNode* input)
{
    graph::NNGraph::TensorNode* w1 = fc1().weight_tensor();
    graph::NNGraph::TensorNode* hidden =
        graph::gemm(w1, input, 1.0, true, false, 1, 0);
    hidden->set_name(tensor_name("fc1_out"));
    hidden = activation().forward(hidden);
    graph::NNGraph::TensorNode* w2 = fc2().weight_tensor();
    graph::NNGraph::TensorNode* out =
        graph::gemm(w2, hidden, 1.0, true, false, 1, 0);
    out->set_name(tensor_name("mlp_out"));
    return out;
}

std::string GptneoMLP::repr() const
{
    return "GptneoMLP(" + graph::module::Mlp::repr() + ")";
}

} // namespace nntile::model::gptneo
