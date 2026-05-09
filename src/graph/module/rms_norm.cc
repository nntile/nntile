/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/rms_norm.cc
 * RMSNorm module implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/module/rms_norm.hh"

#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/nn/ops/rms_norm.hh"

#include <stdexcept>

namespace nntile::graph::module
{

RMSNorm::RMSNorm(graph::NNGraph *graph,
    const std::string &name,
    Index normalized_shape,
    Index axis,
    float eps,
    int redux,
    graph::DataType dtype) :
    Module(graph, name),
    normalized_shape_(normalized_shape),
    axis_(axis),
    eps_(eps),
    redux_(redux),
    dtype_(dtype)
{
    gamma_tensor_ = graph_->tensor({normalized_shape}, dtype_, true);
    gamma_tensor_->set_name(tensor_name("gamma"));
    register_parameter("gamma", gamma_tensor_);
}

graph::NNGraph::TensorNode *RMSNorm::forward(graph::NNGraph::TensorNode *x)
{
    if (x == nullptr)
    {
        throw std::invalid_argument(
            "RMSNorm::forward: input tensor must be non-null");
    }
    return graph::rms_norm(x, gamma_tensor_, axis_, eps_, redux_)
        ->set_name(tensor_name("out"));
}

std::string RMSNorm::repr() const
{
    return "RMSNorm(normalized_shape=" + std::to_string(normalized_shape_) +
           ", eps=" + std::to_string(eps_) + ")";
}

} // namespace nntile::graph::module
