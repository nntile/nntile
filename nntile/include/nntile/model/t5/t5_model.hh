/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/t5/t5_model.hh
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
#include <nntile/model/t5/t5_block.hh>
#include <nntile/model/t5/t5_config.hh>
#include <nntile/module/embedding.hh>
#include <nntile/module/module.hh>
#include <nntile/module/rms_norm.hh>

namespace nntile::model::t5
{

//! T5Model - shared embed + encoder stack + decoder stack
class T5Model : public module::Module
{
private:
    module::Embedding embed_tokens_;
    module::RMSNorm encoder_final_norm_;
    module::RMSNorm decoder_final_norm_;

    std::vector<std::unique_ptr<T5EncoderBlock>> encoder_layers_;
    std::vector<std::unique_ptr<T5DecoderBlock>> decoder_layers_;

    T5Config config_;
    DataType dtype_;

public:
    T5Model(NNGraph* graph,
            const std::string& name,
            const T5Config& config,
            DataType dtype = DataType::FP32);

    //! Forward pass
    //! @param encoder_input_ids (enc_seq, batch) INT64
    //! @param decoder_input_ids (dec_seq, batch) INT64
    //! @param encoder_attention_mask Optional (enc_seq, enc_seq) or nullptr
    //! @param decoder_attention_mask Optional causal mask (dec_seq, dec_seq)
    //! @param cross_attention_mask Optional (enc_seq, dec_seq) or nullptr
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* encoder_input_ids,
        NNGraph::TensorNode* decoder_input_ids,
        NNGraph::TensorNode* encoder_attention_mask = nullptr,
        NNGraph::TensorNode* decoder_attention_mask = nullptr,
        NNGraph::TensorNode* cross_attention_mask = nullptr);

    std::string repr() const override;

    NNGraph::TensorNode* embed_vocab_tensor() const
    {
        return embed_tokens_.vocab_tensor();
    }

    Index num_encoder_layers() const { return config_.num_layers; }
    Index num_decoder_layers() const { return config_.num_decoder_layers; }
};

} // namespace nntile::model::t5
