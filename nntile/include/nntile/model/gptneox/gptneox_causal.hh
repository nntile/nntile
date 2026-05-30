/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/gptneox/gptneox_causal.hh
 * GptneoxCausal - GptneoxModel + lm_head for causal language modeling.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/gptneox/gptneox_config.hh>
#include <nntile/model/gptneox/gptneox_model.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/module.hh>

namespace nntile::model::gptneox
{

//! GptneoxCausal - GptneoxModel + lm_head for next-token prediction
class GptneoxCausal : public module::Module
{
private:
    std::unique_ptr<GptneoxModel> model_;
    module::Linear lm_head_;

    GptneoxConfig config_;
    DataType dtype_;

public:
    //! Constructor
    GptneoxCausal(NNGraph* graph,
                  const std::string& name,
                  const GptneoxConfig& config,
                  DataType dtype = DataType::FP32);

    //! Forward pass
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input_ids,
        NNGraph::TensorNode* sin = nullptr,
        NNGraph::TensorNode* cos = nullptr,
        NNGraph::TensorNode* mask = nullptr);

    std::string repr() const override;

    GptneoxModel* model() { return model_.get(); }
};

} // namespace nntile::model::gptneox
