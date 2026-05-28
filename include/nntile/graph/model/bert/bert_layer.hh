/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/bert/bert_layer.hh
 * BertLayer - attention + feed-forward sublayers.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/graph/model/bert/bert_attention.hh>
#include <nntile/graph/model/bert/bert_config.hh>
#include <nntile/graph/model/bert/bert_intermediate.hh>
#include <nntile/graph/model/bert/bert_output.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::graph::model::bert
{

class BertLayer : public graph::module::Module
{
private:
    BertAttention attention_;
    BertIntermediate intermediate_;
    BertOutput output_;

    BertConfig config_;
    graph::DataType dtype_;

public:
    BertLayer(graph::NNGraph* graph,
              const std::string& name,
              const BertConfig& config,
              graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x,
        graph::NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;
};

} // namespace nntile::graph::model::bert
