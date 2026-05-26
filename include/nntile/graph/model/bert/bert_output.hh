/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/bert/bert_output.hh
 * BertOutput - FFN output projection, residual, layer norm.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/graph/model/bert/bert_config.hh>
#include <nntile/graph/module/layer_norm.hh>
#include <nntile/graph/module/linear.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::model::bert
{

class BertOutput : public graph::module::Module
{
private:
    graph::module::Linear dense_;
    graph::module::LayerNorm layer_norm_;

    BertConfig config_;
    graph::DataType dtype_;

public:
    BertOutput(graph::NNGraph* graph,
               const std::string& name,
               const BertConfig& config,
               graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* hidden,
        graph::NNGraph::TensorNode* residual);

    std::string repr() const override;
};

} // namespace nntile::model::bert
