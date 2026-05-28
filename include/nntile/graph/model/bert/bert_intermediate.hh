/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/bert/bert_intermediate.hh
 * BertIntermediate - Linear + activation from BertConfig::hidden_act.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/graph/model/bert/bert_config.hh>
#include <nntile/graph/module/activation.hh>
#include <nntile/graph/module/linear.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::graph::model::bert
{

class BertIntermediate : public graph::module::Module
{
private:
    graph::module::Linear dense_;
    graph::module::Activation activation_;

    BertConfig config_;
    graph::DataType dtype_;

public:
    BertIntermediate(graph::NNGraph* graph,
                     const std::string& name,
                     const BertConfig& config,
                     graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(graph::NNGraph::TensorNode* x);

    std::string repr() const override;
};

} // namespace nntile::graph::model::bert
