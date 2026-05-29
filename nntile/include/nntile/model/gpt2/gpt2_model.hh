/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/gpt2/gpt2_model.hh
 * GPT2Model - wte + wpe + add -> decoder layers + final norm.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <string>
#include <vector>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/gpt2/gpt2_block.hh>
#include <nntile/model/gpt2/gpt2_config.hh>
#include <nntile/module/embedding.hh>
#include <nntile/module/module.hh>
#include <nntile/module/layer_norm.hh>

namespace nntile::model::gpt2
{

//! GPT2Model - wte + wpe + add -> num_hidden_layers x Gpt2Block + ln_f
class Gpt2Model : public graph::module::Module
{
private:
    graph::module::Embedding wte_;
    graph::module::Embedding wpe_;
    std::vector<std::unique_ptr<Gpt2Block>> layers_;
    graph::module::LayerNorm ln_f_;

    Gpt2Config config_;
    graph::DataType dtype_;

public:
    Gpt2Model(graph::NNGraph* graph,
              const std::string& name,
              const Gpt2Config& config,
              graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input_ids,
        graph::NNGraph::TensorNode* position_ids,
        graph::NNGraph::TensorNode* mask = nullptr,
        bool causal = false);

    std::string repr() const override;

    Index num_layers() const { return config_.num_hidden_layers; }

    graph::NNGraph::TensorNode* wte_vocab_tensor() const
    {
        return wte_.vocab_tensor();
    }
};

} // namespace nntile::model::gpt2
