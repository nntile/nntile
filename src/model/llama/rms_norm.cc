/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/model/llama/rms_norm.cc
 * RMSNorm module implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/llama/rms_norm.hh"
#include "nntile/graph/nn/rms_norm.hh"

#include <stdexcept>

namespace nntile::model::llama
{

RMSNorm::RMSNorm(graph::NNGraph* graph,
                 const std::string& name,
                 Index normalized_shape,
                 Index axis,
                 float eps,
                 int redux,
                 graph::DataType dtype)
    : module::Module(graph, name)
    , normalized_shape_(normalized_shape)
    , axis_(axis)
    , eps_(eps)
    , redux_(redux)
    , dtype_(dtype)
{
    gamma_tensor_ = graph_->tensor(
        {normalized_shape},
        tensor_name("gamma"),
        dtype_,
        true);
    register_parameter("gamma", gamma_tensor_);
}

graph::NNGraph::TensorNode* RMSNorm::forward(
    graph::NNGraph::TensorNode* x)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "RMSNorm::forward: input tensor must be non-null");
    }
    return graph::rms_norm(x, gamma_tensor_, tensor_name("out"),
                           axis_, eps_, redux_);
}

std::string RMSNorm::repr() const
{
    return "RMSNorm(normalized_shape=" + std::to_string(normalized_shape_) +
           ", eps=" + std::to_string(eps_) + ")";
}

} // namespace nntile::model::llama
