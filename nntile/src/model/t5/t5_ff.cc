/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/t5/t5_ff.cc
 * T5LayerFF implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/t5/t5_ff.hh"
#include "nntile/nn/ops/add.hh"

#include <stdexcept>

namespace nntile::model::t5
{

T5LayerFF::T5LayerFF(NNGraph* graph,
                     const std::string& name,
                     const T5Config& config,
                     DataType dtype)
    : module::Module(graph, name)
    , layer_norm_(graph, name + "_layer_norm",
                  config.d_model, 2, config.layer_norm_epsilon, 0,
                  dtype) // axis=2 for (batch, seq, d_model)
    , dense_(graph, name + "_dense",
             config.d_model,
             config.d_ff,
             config.d_model,
             module::ActivationType::GELUTANH,
             dtype)
    , config_(config)
    , dtype_(dtype)
{
    register_module("layer_norm", &layer_norm_);
    register_module("dense", &dense_);
}

NNGraph::TensorNode* T5LayerFF::forward(
    NNGraph::TensorNode* input)
{
    if(input == nullptr)
    {
        throw std::invalid_argument(
            "T5LayerFF::forward: input tensor must be non-null");
    }

    // layer_norm on (batch, seq, d_model)
    NNGraph::TensorNode* x_norm = layer_norm_.forward(input);

    // GatedMlp on (batch, seq, d_model) layout
    NNGraph::TensorNode* ff_out = dense_.forward(x_norm);
    ff_out->set_name(tensor_name("ff_out"));

    return add(1.0, input, 1.0, ff_out);
}

std::string T5LayerFF::repr() const
{
    return "T5LayerFF(d_model=" + std::to_string(config_.d_model) +
           ", d_ff=" + std::to_string(config_.d_ff) + ")";
}

} // namespace nntile::model::t5
