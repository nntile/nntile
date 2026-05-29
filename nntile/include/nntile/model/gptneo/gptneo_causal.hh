/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/gptneo/gptneo_causal.hh
 * GptneoCausal - GptneoModel + lm_head for causal language modeling.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/gptneo/gptneo_config.hh>
#include <nntile/model/gptneo/gptneo_model.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/module.hh>

namespace nntile::model::gptneo
{

//! GptneoCausal - GptneoModel + lm_head for next-token prediction
class GptneoCausal : public module::Module
{
private:
    std::unique_ptr<GptneoModel> model_;
    module::Linear lm_head_;

    GptneoConfig config_;
    DataType dtype_;

public:
    //! Constructor
    GptneoCausal(NNGraph* graph,
                 const std::string& name,
                 const GptneoConfig& config,
                 DataType dtype = DataType::FP32);

    //! Forward pass
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input_ids,
        NNGraph::TensorNode* position_ids,
        NNGraph::TensorNode* mask = nullptr,
        NNGraph::TensorNode* local_mask = nullptr);

    std::string repr() const override;

    GptneoModel* model() { return model_.get(); }
};

} // namespace nntile::model::gptneo
