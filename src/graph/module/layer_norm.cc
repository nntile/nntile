/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/module/layer_norm.cc
 * LayerNorm module implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/module/layer_norm.hh"
#include "nntile/graph/nn/layer_norm.hh"

#include <stdexcept>

#include "nntile/graph/io/safetensors.hh"

namespace nntile::graph::module
{

LayerNorm::LayerNorm(graph::NNGraph* graph,
                     const std::string& name,
                     Index normalized_shape,
                     Index axis,
                     float eps,
                     int redux,
                     graph::DataType dtype)
    : Module(graph, name)
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
    beta_tensor_ = graph_->tensor(
        {normalized_shape},
        tensor_name("beta"),
        dtype_,
        true);
    register_parameter("gamma", gamma_tensor_);
    register_parameter("beta", beta_tensor_);
}

graph::NNGraph::TensorNode* LayerNorm::forward(
    graph::NNGraph::TensorNode* x)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "LayerNorm::forward: input tensor must be non-null");
    }
    return graph::layer_norm(x, gamma_tensor_, beta_tensor_,
                             tensor_name("out"), axis_, eps_, redux_);
}

std::string LayerNorm::repr() const
{
    return "LayerNorm(normalized_shape=" + std::to_string(normalized_shape_) +
           ", eps=" + std::to_string(eps_) + ")";
}

void LayerNorm::import_hf(const graph::io::SafeTensorsReader& reader,
                          const std::string& hf_prefix)
{
    const std::string gamma_key =
        hf_prefix.empty() ? "weight" : hf_prefix + ".weight";
    const std::string beta_key =
        hf_prefix.empty() ? "bias" : hf_prefix + ".bias";

    if(!reader.has_tensor(gamma_key))
    {
        throw std::runtime_error(
            "LayerNorm::import_hf: '" + gamma_key + "' not found");
    }
    const auto& gamma_info = reader.tensor_info(gamma_key);
    if(gamma_info.shape.size() != 1 ||
       gamma_info.shape[0] != static_cast<std::int64_t>(normalized_shape_))
    {
        throw std::runtime_error(
            "LayerNorm::import_hf: shape mismatch for " + gamma_key);
    }
    auto gamma_data = reader.read_tensor(gamma_key);
    gamma_tensor_->data()->set_bind_hint(std::move(gamma_data));
    gamma_tensor_->mark_input(true);

    if(!reader.has_tensor(beta_key))
    {
        throw std::runtime_error(
            "LayerNorm::import_hf: '" + beta_key + "' not found");
    }
    const auto& beta_info = reader.tensor_info(beta_key);
    if(beta_info.shape.size() != 1 ||
       beta_info.shape[0] != static_cast<std::int64_t>(normalized_shape_))
    {
        throw std::runtime_error(
            "LayerNorm::import_hf: shape mismatch for " + beta_key);
    }
    auto beta_data = reader.read_tensor(beta_key);
    beta_tensor_->data()->set_bind_hint(std::move(beta_data));
    beta_tensor_->mark_input(true);
}

void LayerNorm::export_hf(graph::io::SafeTensorsWriter& writer,
                          const std::string& hf_prefix) const
{
    const std::string gamma_key =
        hf_prefix.empty() ? "weight" : hf_prefix + ".weight";
    const std::string beta_key =
        hf_prefix.empty() ? "bias" : hf_prefix + ".bias";

    const auto* gamma_hint = gamma_tensor_->data()->get_bind_hint();
    if(gamma_hint == nullptr)
    {
        throw std::runtime_error(
            "LayerNorm::export_hf: gamma has no data");
    }
    writer.add_tensor(gamma_key, dtype_,
                      {static_cast<std::int64_t>(normalized_shape_)},
                      *gamma_hint);

    const auto* beta_hint = beta_tensor_->data()->get_bind_hint();
    if(beta_hint == nullptr)
    {
        throw std::runtime_error(
            "LayerNorm::export_hf: beta has no data");
    }
    writer.add_tensor(beta_key, dtype_,
                      {static_cast<std::int64_t>(normalized_shape_)},
                      *beta_hint);
}

} // namespace nntile::graph::module
