/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/dataset/seq2seq_lm_mmap.hh
 * Memory-mapped uint16 stream and seq2seq LM batch iterator (Fortran
 * ``(seq, batch)`` layout for ``Runtime::bind_data``).
 *
 * Each training window uses ``enc_seq + dec_seq`` consecutive tokens from
 * ``train.bin``. Encoder input is the first ``enc_seq`` ids; decoder labels
 * are the last ``dec_seq`` ids. Decoder input is teacher-forced: position 0
 * is ``decoder_start_token_id``, then previous label tokens.
 *
 * Graph ``cross_entropy`` takes logits ``(vocab, dec_seq, batch)`` and labels
 * ``(dec_seq, batch)``.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/dataset/causal_lm_mmap.hh>

namespace nntile::dataset
{

//! One minibatch for encoder-decoder training.
struct Seq2SeqLmBatch
{
    std::vector<std::int64_t> encoder_input_ids;
    std::vector<std::int64_t> decoder_input_ids;
    std::vector<std::int64_t> labels;
};

//! Iterator parameters (encoder/decoder lengths, batch, shuffle).
struct Seq2SeqLmBatchConfig
{
    Index n_enc_seq = 8;
    Index n_dec_seq = 8;
    Index n_batch = 2;
    bool shuffle = false;
    unsigned seed = 42;
    std::int64_t decoder_start_token_id = 0;
};

//! Yields batches from a mmap ``uint16`` stream (reuses ``TokenMemoryMap``).
class Seq2SeqLmBatchIterator
{
public:
    //! ``vocab_size``: if ``> 0``, ``next`` throws if any token is out of
    //! range.
    Seq2SeqLmBatchIterator(
        TokenMemoryMap const& tokens,
        Seq2SeqLmBatchConfig const& cfg,
        Index vocab_size);

    //! Fills ``batch``; sizes are ``enc_seq * batch``, ``dec_seq * batch``.
    //! Returns ``false`` when no full batch remains.
    bool next(Seq2SeqLmBatch& batch);

    std::size_t num_batches() const noexcept;
    std::size_t batch_index() const noexcept;

private:
    TokenMemoryMap const* tokens_;
    Seq2SeqLmBatchConfig cfg_;
    Index vocab_size_;
    std::size_t num_seq_ = 0;
    std::size_t num_batches_ = 0;
    std::size_t batch_idx_ = 0;
    std::vector<std::size_t> seq_order_;
};

} // namespace nntile::dataset
