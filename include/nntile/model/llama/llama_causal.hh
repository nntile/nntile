/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/llama/llama_causal.hh
 * LlamaCausal - LlamaModel + lm_head for causal language modeling.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/llama/llama_config.hh>
#include <nntile/model/llama/llama_model.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/module.hh>

namespace nntile::model::llama
{

//! LlamaCausal - LlamaModel + lm_head for next-token prediction
class LlamaCausal : public module::Module
{
private:
    std::unique_ptr<LlamaModel> model_;
    module::Linear lm_head_;

    LlamaConfig config_;
    graph::DataType dtype_;

public:
    //! Constructor
    LlamaCausal(graph::NNGraph* graph,
                const std::string& name,
                const LlamaConfig& config,
                graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input_ids,
        graph::NNGraph::TensorNode* sin = nullptr,
        graph::NNGraph::TensorNode* cos = nullptr,
        graph::NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    LlamaModel* model() { return model_.get(); }
};

} // namespace nntile::model::llama
