/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/bert/bert_output.hh
 * BertOutput - FFN output projection, residual, layer norm.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph.hh>
#include <nntile/model/bert/bert_config.hh>
#include <nntile/module/layer_norm.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/module.hh>

namespace nntile::model::bert
{

class BertOutput : public module::Module
{
private:
    module::Linear dense_;
    module::LayerNorm layer_norm_;

    BertConfig config_;
    DataType dtype_;

public:
    BertOutput(NNGraph* graph,
               const std::string& name,
               const BertConfig& config,
               DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* hidden,
        NNGraph::TensorNode* residual);

    std::string repr() const override;
};

} // namespace nntile::model::bert
