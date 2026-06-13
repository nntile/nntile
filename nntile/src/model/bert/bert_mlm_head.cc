/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/bert/bert_mlm_head.cc
 * BertMlmHead implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/bert/bert_mlm_head.hh"
#include "nntile/model/bert/bert_config.hh"

namespace nntile::model::bert
{

namespace
{

NNGraph::TensorNode* require_tied_word_vocab(
    NNGraph::TensorNode* tied_word_vocab)
{
    if(tied_word_vocab == nullptr)
    {
        throw std::invalid_argument(
            "BertMlmHead: tied_word_vocab must be non-null");
    }
    return tied_word_vocab;
}

} // anonymous namespace

BertMlmHead::BertMlmHead(NNGraph* graph,
                         const std::string& name,
                         const BertConfig& config,
                         NNGraph::TensorNode* tied_word_vocab,
                         DataType dtype)
    : module::Module(graph, name)
    , transform_dense_(graph, name + "_transform_dense",
                       config.hidden_size,
                       config.hidden_size,
                       true,
                       dtype)
    , transform_act_(graph, name + "_transform_act",
                     activation_type_from_config(config))
    , transform_ln_(graph, name + "_transform_ln",
                    config.hidden_size, -1, config.layer_norm_eps, 0, dtype)
    , decoder_(graph,
               name + "_decoder",
               require_tied_word_vocab(tied_word_vocab),
               graph_->tensor({config.vocab_size}, dtype, true))
    , config_(config)
    , dtype_(dtype)
{
    config_.validate();
    register_module("transform_dense", &transform_dense_);
    register_module("transform_act", &transform_act_);
    register_module("transform_ln", &transform_ln_);
    register_module("decoder", &decoder_);
}

NNGraph::TensorNode* BertMlmHead::forward(
    NNGraph::TensorNode* hidden)
{
    NNGraph::TensorNode* t = transform_dense_.forward(hidden);
    t = transform_act_.forward(t);
    t = transform_ln_.forward(t);

    NNGraph::TensorNode* logits = decoder_.forward(t);
    return logits;
}

std::string BertMlmHead::repr() const
{
    return "BertMlmHead(vocab=" + std::to_string(config_.vocab_size) + ")";
}

} // namespace nntile::model::bert
