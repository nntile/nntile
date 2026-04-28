/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gpt2/gpt2_mlp.cc
 * GPT2MLP implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gpt2/gpt2_mlp.hh"
#include "nntile/graph/nn/transpose.hh"

namespace nntile::model::gpt2
{

Gpt2MLP::Gpt2MLP(graph::NNGraph* graph,
                 const std::string& name,
                 const Gpt2Config& config,
                 graph::DataType dtype)
    : graph::module::Mlp(graph, name,
                         config.hidden_size,
                         config.intermediate_size,
                         config.hidden_size,
                         graph::module::ActivationType::GELU,
                         dtype)
{
}

graph::NNGraph::TensorNode* Gpt2MLP::forward(
    graph::NNGraph::TensorNode* input)
{
    // Transpose (hidden, seq, batch) -> (seq, batch, hidden) for Mlp (ndim=1)
    graph::NNGraph::TensorNode* x =
        graph::transpose(input, tensor_name("x"), 1);
    graph::NNGraph::TensorNode* out = graph::module::Mlp::forward(x);
    // Transpose back to (hidden, seq, batch)
    return graph::transpose(out, tensor_name("mlp_out"), 2);
}

std::string Gpt2MLP::repr() const
{
    return "Gpt2MLP(" + graph::module::Mlp::repr() + ")";
}

} // namespace nntile::model::gpt2
