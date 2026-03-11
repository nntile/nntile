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

#include "nntile/graph/model/llama/llama_mlp.hh"
#include "nntile/graph/nn/transpose.hh"

namespace nntile::model::llama
{

LlamaMLP::LlamaMLP(graph::NNGraph* graph,
                   const std::string& name,
                   const LlamaConfig& config,
                   graph::DataType dtype)
    : graph::module::GatedMlp(graph, name,
                       config.hidden_size,
                       config.intermediate_size,
                       config.hidden_size,
                       graph::module::ActivationType::SILU,
                       dtype)
{
}

graph::NNGraph::TensorNode* LlamaMLP::forward(
    graph::NNGraph::TensorNode* input)
{
    // Transpose (hidden, seq, batch) -> (seq, batch, hidden) for GatedMlp (ndim=1)
    graph::NNGraph::TensorNode* x =
        graph::transpose(input, tensor_name("x"), 1);
    graph::NNGraph::TensorNode* out = graph::module::GatedMlp::forward(x);
    // Transpose back to (hidden, seq, batch) (ndim=2)
    return graph::transpose(out, tensor_name("mlp_out"), 2);
}

std::string LlamaMLP::repr() const
{
    return "LlamaMLP(" + graph::module::GatedMlp::repr() + ")";
}

} // namespace nntile::model::llama
