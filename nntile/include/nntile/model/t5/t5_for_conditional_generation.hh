/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/t5/t5_for_conditional_generation.hh
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
#include <nntile/model/t5/t5_config.hh>
#include <nntile/model/t5/t5_model.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/module.hh>

namespace nntile::model::t5
{

//! T5ForConditionalGeneration - encoder + decoder + lm_head
class T5ForConditionalGeneration : public module::Module
{
private:
    std::unique_ptr<T5Model> model_;
    module::Linear lm_head_;

    T5Config config_;
    DataType dtype_;

public:
    T5ForConditionalGeneration(NNGraph* graph,
                               const std::string& name,
                               const T5Config& config,
                               DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* encoder_input_ids,
        NNGraph::TensorNode* decoder_input_ids,
        NNGraph::TensorNode* encoder_attention_mask = nullptr,
        NNGraph::TensorNode* decoder_attention_mask = nullptr,
        NNGraph::TensorNode* cross_attention_mask = nullptr);

    std::string repr() const override;

    T5Model* model() { return model_.get(); }
};

} // namespace nntile::model::t5
