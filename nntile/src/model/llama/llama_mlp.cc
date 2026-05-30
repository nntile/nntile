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

#include "nntile/nn/ops/transpose.hh"

namespace nntile::model::llama
{

LlamaMLP::LlamaMLP(NNGraph *graph,
    const std::string &name,
    const LlamaConfig &config,
    DataType dtype) :
    module::GatedMlp(graph,
        name,
        config.hidden_size,
        config.intermediate_size,
        config.hidden_size,
        module::ActivationType::SILU,
        dtype)
{
}

NNGraph::TensorNode *LlamaMLP::forward(
    NNGraph::TensorNode *input)
{
    // Transpose (hidden, seq, batch) -> (seq, batch, hidden) for GatedMlp
    // (ndim=1)
    NNGraph::TensorNode *x = transpose(input, 1);
    x->set_name(tensor_name("x"));
    NNGraph::TensorNode *out = module::GatedMlp::forward(x);
    NNGraph::TensorNode *mlp_out = transpose(out, 2);
    mlp_out->set_name(tensor_name("mlp_out"));
    return mlp_out;
}

std::string LlamaMLP::repr() const
{
    return "LlamaMLP(" + module::GatedMlp::repr() + ")";
}

} // namespace nntile::model::llama
