/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/t5/t5_model.hh
 * T5Model - encoder + decoder (embedding shared).
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
#include <nntile/graph/model/t5/t5_block.hh>
#include <nntile/graph/model/t5/t5_config.hh>
#include <nntile/graph/module/embedding.hh>
#include <nntile/graph/module/module.hh>
#include <nntile/graph/module/rms_norm.hh>

namespace nntile::model::t5
{

//! T5Model - shared embed + encoder stack + decoder stack
class T5Model : public graph::module::Module
{
private:
    graph::module::Embedding embed_tokens_;
    graph::module::RMSNorm encoder_final_norm_;
    graph::module::RMSNorm decoder_final_norm_;

    std::vector<std::unique_ptr<T5EncoderBlock>> encoder_layers_;
    std::vector<std::unique_ptr<T5DecoderBlock>> decoder_layers_;

    T5Config config_;
    graph::DataType dtype_;

public:
    T5Model(graph::NNGraph* graph,
            const std::string& name,
            const T5Config& config,
            graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    //! @param encoder_input_ids (enc_seq, batch) INT64
    //! @param decoder_input_ids (dec_seq, batch) INT64
    //! @param encoder_attention_mask Optional (enc_seq, enc_seq) or nullptr
    //! @param decoder_attention_mask Optional causal mask (dec_seq, dec_seq)
    //! @param cross_attention_mask Optional (dec_seq, enc_seq) or nullptr
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* encoder_input_ids,
        graph::NNGraph::TensorNode* decoder_input_ids,
        graph::NNGraph::TensorNode* encoder_attention_mask = nullptr,
        graph::NNGraph::TensorNode* decoder_attention_mask = nullptr,
        graph::NNGraph::TensorNode* cross_attention_mask = nullptr);

    std::string repr() const override;

    Index num_encoder_layers() const { return config_.num_layers; }
    Index num_decoder_layers() const { return config_.num_decoder_layers; }
};

} // namespace nntile::model::t5
