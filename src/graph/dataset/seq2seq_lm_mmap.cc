/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/dataset/seq2seq_lm_mmap.cc
 * Mmap token stream and seq2seq LM batch iterator.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/dataset/seq2seq_lm_mmap.hh"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace nntile::graph::dataset
{

Seq2SeqLmBatchIterator::Seq2SeqLmBatchIterator(
    TokenMemoryMap const& tokens,
    Seq2SeqLmBatchConfig const& cfg,
    Index vocab_size)
    : tokens_(&tokens)
    , cfg_(cfg)
    , vocab_size_(vocab_size)
{
    if(cfg_.n_enc_seq <= 0 || cfg_.n_dec_seq <= 0 || cfg_.n_batch <= 0)
    {
        throw std::invalid_argument(
            "Seq2SeqLmBatchIterator: enc/dec seq and batch must be positive");
    }
    const std::size_t window =
        static_cast<std::size_t>(cfg_.n_enc_seq) +
        static_cast<std::size_t>(cfg_.n_dec_seq);
    const std::size_t ntok = tokens_->num_tokens();
    num_seq_ = ntok / window;
    num_batches_ = num_seq_ / static_cast<std::size_t>(cfg_.n_batch);
    seq_order_.resize(num_seq_);
    for(std::size_t i = 0; i < num_seq_; ++i)
    {
        seq_order_[i] = i;
    }
    if(cfg_.shuffle && num_seq_ > 1)
    {
        std::mt19937 gen(cfg_.seed);
        std::shuffle(seq_order_.begin(), seq_order_.end(), gen);
    }
}

bool Seq2SeqLmBatchIterator::next(Seq2SeqLmBatch& batch)
{
    if(batch_idx_ >= num_batches_)
    {
        return false;
    }
    const Index n_enc = cfg_.n_enc_seq;
    const Index n_dec = cfg_.n_dec_seq;
    const Index n_batch = cfg_.n_batch;
    const std::size_t enc_nelem =
        static_cast<std::size_t>(n_enc) * static_cast<std::size_t>(n_batch);
    const std::size_t dec_nelem =
        static_cast<std::size_t>(n_dec) * static_cast<std::size_t>(n_batch);
    batch.encoder_input_ids.resize(enc_nelem);
    batch.decoder_input_ids.resize(dec_nelem);
    batch.labels.resize(dec_nelem);

    const std::size_t window =
        static_cast<std::size_t>(n_enc) + static_cast<std::size_t>(n_dec);
    std::uint16_t const* const tok = tokens_->data();

    for(Index b = 0; b < n_batch; ++b)
    {
        const std::size_t seq_slot =
            seq_order_[batch_idx_ * static_cast<std::size_t>(n_batch) + b];
        const std::size_t off = seq_slot * window;

        for(Index s = 0; s < n_enc; ++s)
        {
            const std::uint16_t ti =
                tok[off + static_cast<std::size_t>(s)];
            if(vocab_size_ > 0 &&
                static_cast<Index>(ti) >= vocab_size_)
            {
                throw std::runtime_error(
                    "Seq2SeqLmBatchIterator: token id >= vocab_size");
            }
            const std::size_t idx =
                static_cast<std::size_t>(s) +
                static_cast<std::size_t>(n_enc) * static_cast<std::size_t>(b);
            batch.encoder_input_ids[idx] = static_cast<std::int64_t>(ti);
        }

        for(Index s = 0; s < n_dec; ++s)
        {
            const std::uint16_t tt =
                tok[off + static_cast<std::size_t>(n_enc) +
                    static_cast<std::size_t>(s)];
            if(vocab_size_ > 0 &&
                static_cast<Index>(tt) >= vocab_size_)
            {
                throw std::runtime_error(
                    "Seq2SeqLmBatchIterator: token id >= vocab_size");
            }
            const std::size_t idx =
                static_cast<std::size_t>(s) +
                static_cast<std::size_t>(n_dec) * static_cast<std::size_t>(b);
            batch.labels[idx] = static_cast<std::int64_t>(tt);
            if(s == 0)
            {
                batch.decoder_input_ids[idx] = cfg_.decoder_start_token_id;
            }
            else
            {
                batch.decoder_input_ids[idx] = batch.labels[idx - 1];
            }
        }
    }
    ++batch_idx_;
    return true;
}

std::size_t Seq2SeqLmBatchIterator::num_batches() const noexcept
{
    return num_batches_;
}

std::size_t Seq2SeqLmBatchIterator::batch_index() const noexcept
{
    return batch_idx_;
}

} // namespace nntile::graph::dataset
