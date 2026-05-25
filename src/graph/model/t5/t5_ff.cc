/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/t5/t5_ff.cc
 * T5LayerFF implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/t5/t5_ff.hh"
#include "nntile/graph/nn/add.hh"
#include "nntile/graph/nn/transpose.hh"

#include <stdexcept>

namespace nntile::model::t5
{

T5LayerFF::T5LayerFF(graph::NNGraph* graph,
                     const std::string& name,
                     const T5Config& config,
                     graph::DataType dtype)
    : graph::module::Module(graph, name)
    , layer_norm_(graph, name + "_layer_norm",
                  config.d_model, 0, config.layer_norm_epsilon, 0, dtype)
    , dense_(graph, name + "_dense",
             config.d_model,
             config.d_ff,
             config.d_model,
             graph::module::ActivationType::GELU,
             dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("layer_norm", &layer_norm_);
    register_module("dense", &dense_);
}

graph::NNGraph::TensorNode* T5LayerFF::forward(
    graph::NNGraph::TensorNode* input)
{
    if(input == nullptr)
    {
        throw std::invalid_argument(
            "T5LayerFF::forward: input tensor must be non-null");
    }

    // layer_norm
    graph::NNGraph::TensorNode* x_norm = layer_norm_.forward(input);

    // Transpose (d_model, seq, batch) -> (seq, batch, d_model) for GatedMlp
    graph::NNGraph::TensorNode* x_t =
        graph::transpose(x_norm, tensor_name("x_t"), 1);
    graph::NNGraph::TensorNode* ff_out = dense_.forward(x_t);
    // Transpose back to (d_model, seq, batch)
    graph::NNGraph::TensorNode* ff_out_t =
        graph::transpose(ff_out, tensor_name("ff_out_t"), 2);

    // Residual: input + ff_out
    return graph::add(1.0, input, 1.0, ff_out_t, tensor_name("ff_residual"));
}

std::string T5LayerFF::repr() const
{
    return "T5LayerFF(d_model=" + std::to_string(config_.d_model) +
           ", d_ff=" + std::to_string(config_.d_ff) + ")";
}

} // namespace nntile::model::t5
