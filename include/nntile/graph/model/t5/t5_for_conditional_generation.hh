/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/t5/t5_for_conditional_generation.hh
 * T5ForConditionalGeneration - T5Model + lm_head for seq2seq.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/model/t5/t5_config.hh>
#include <nntile/graph/model/t5/t5_model.hh>
#include <nntile/graph/module/linear.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::model::t5
{

//! T5ForConditionalGeneration - encoder + decoder + lm_head
class T5ForConditionalGeneration : public graph::module::Module
{
private:
    std::unique_ptr<T5Model> model_;
    graph::module::Linear lm_head_;

    T5Config config_;
    graph::DataType dtype_;

public:
    T5ForConditionalGeneration(graph::NNGraph* graph,
                               const std::string& name,
                               const T5Config& config,
                               graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* encoder_input_ids,
        graph::NNGraph::TensorNode* decoder_input_ids,
        graph::NNGraph::TensorNode* encoder_attention_mask = nullptr,
        graph::NNGraph::TensorNode* decoder_attention_mask = nullptr,
        graph::NNGraph::TensorNode* cross_attention_mask = nullptr);

    std::string repr() const override;

    T5Model* model() { return model_.get(); }
};

} // namespace nntile::model::t5
