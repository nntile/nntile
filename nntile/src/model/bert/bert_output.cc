/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/bert/bert_output.cc
 * BertOutput implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/bert/bert_output.hh"
#include "nntile/nn/ops/add.hh"

#include <stdexcept>

namespace nntile::model::bert
{

BertOutput::BertOutput(NNGraph* graph,
                       const std::string& name,
                       const BertConfig& config,
                       DataType dtype)
    : module::Module(graph, name)
    , dense_(graph, name + "_dense",
             config.intermediate_size,
             config.hidden_size,
             true,
             dtype)
    , layer_norm_(graph, name + "_ln",
                  config.hidden_size, 2, config.layer_norm_eps, 0, dtype)
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("dense", &dense_);
    register_module("ln", &layer_norm_);
}

NNGraph::TensorNode* BertOutput::forward(
    NNGraph::TensorNode* hidden,
    NNGraph::TensorNode* residual)
{
    if(hidden == nullptr || residual == nullptr)
    {
        throw std::invalid_argument(
            "BertOutput::forward: hidden and residual must be non-null");
    }

    NNGraph::TensorNode* proj = dense_.forward(hidden);
    NNGraph::TensorNode* summed =
        add(1.0, residual, 1.0, proj);
    return layer_norm_.forward(summed);
}

std::string BertOutput::repr() const
{
    return "BertOutput(hidden=" + std::to_string(config_.hidden_size) + ")";
}

} // namespace nntile::model::bert
