/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gpt2/gpt2_causal.hh
 * Gpt2Causal - GPT2Model + lm_head for causal language modeling.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/model/gpt2/gpt2_config.hh>
#include <nntile/graph/model/gpt2/gpt2_model.hh>
#include <nntile/graph/module/linear.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::model::gpt2
{

//! Gpt2Causal - GPT2Model + lm_head for next-token prediction
class Gpt2Causal : public graph::module::Module
{
private:
    std::unique_ptr<Gpt2Model> model_;
    graph::module::Linear lm_head_;

    Gpt2Config config_;
    graph::DataType dtype_;

public:
    //! Constructor
    Gpt2Causal(graph::NNGraph* graph,
               const std::string& name,
               const Gpt2Config& config,
               graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input_ids,
        graph::NNGraph::TensorNode* position_ids = nullptr,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    Gpt2Model* model() { return model_.get(); }
};

} // namespace nntile::model::gpt2
