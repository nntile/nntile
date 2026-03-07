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
#include "nntile/graph/nn/rms_norm.hh"

#include <stdexcept>

#include "nntile/graph/io/safetensors.hh"

namespace nntile::module
{

RMSNorm::RMSNorm(graph::NNGraph* graph,
                 const std::string& name,
                 Index normalized_shape,
                 Index axis,
                 float eps,
                 int redux,
                 graph::DataType dtype)
    : graph::module::Module(graph, name)
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

void RMSNorm::import_hf(const graph::io::SafeTensorsReader& reader,
                        const std::string& hf_prefix)
{
    const std::string key = hf_prefix.empty() ? "weight" : hf_prefix + ".weight";
    if(!reader.has_tensor(key))
    {
        throw std::runtime_error("RMSNorm::import_hf: '" + key + "' not found");
    }
    const auto& info = reader.tensor_info(key);
    if(info.shape.size() != 1 ||
       info.shape[0] != static_cast<std::int64_t>(normalized_shape_))
    {
        throw std::runtime_error("RMSNorm::import_hf: shape mismatch for " + key);
    }
    auto data = reader.read_tensor(key);
    gamma_tensor_->data()->set_bind_hint(std::move(data));
    gamma_tensor_->mark_input(true);
}

void RMSNorm::export_hf(graph::io::SafeTensorsWriter& writer,
                        const std::string& hf_prefix) const
{
    const std::string key = hf_prefix.empty() ? "weight" : hf_prefix + ".weight";
    const auto* hint = gamma_tensor_->data()->get_bind_hint();
    if(hint == nullptr)
    {
        throw std::runtime_error("RMSNorm::export_hf: gamma has no data");
    }
    writer.add_tensor(key, dtype_,
                      {static_cast<std::int64_t>(normalized_shape_)},
                      *hint);
}

} // namespace nntile::module
