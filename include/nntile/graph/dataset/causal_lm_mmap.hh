/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/dataset/causal_lm_mmap.hh
 * Memory-mapped uint16 token stream and causal LM batch iterator (Fortran
 * ``(seq, batch)`` layout for ``TileGraph::Runtime::bind_data``).
 *
 * Token files match ``wrappers/python/examples/llama_training.py`` (raw
 * ``uint16``, one token per element). Training windows use ``seq_len + 1``
 * tokens per sequence; ``input_ids`` are the first ``seq_len`` tokens and
 * ``target_ids`` are the last ``seq_len`` (next-token shifted). Graph
 * ``cross_entropy`` takes logits ``(vocab, seq, batch)`` and labels
 * ``(seq, batch)`` (same layout as ``target_ids``).
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nntile/base_types.hh>

namespace nntile::graph::dataset
{

//! One minibatch: ``input_ids`` and ``target_ids``, shape ``(n_seq, n_batch)``
//! Fortran, index ``(s, b)`` at ``s + n_seq * b``.
struct CausalLmBatch
{
    std::vector<std::int64_t> input_ids;
    std::vector<std::int64_t> target_ids;
};

//! Iterator parameters (sequence length and batch size; optional shuffle).
struct CausalLmBatchConfig
{
    Index n_seq = 8;
    Index n_batch = 2;
    bool shuffle = false;
    unsigned seed = 42;
};

//! Read-only mmap of a raw ``uint16`` 1-D token file (e.g. ``train.bin``).
class TokenMemoryMap
{
public:
    explicit TokenMemoryMap(std::string path);
    ~TokenMemoryMap();

    TokenMemoryMap(TokenMemoryMap const&) = delete;
    TokenMemoryMap& operator=(TokenMemoryMap const&) = delete;
    TokenMemoryMap(TokenMemoryMap&& other) noexcept;
    TokenMemoryMap& operator=(TokenMemoryMap&& other) noexcept;

    //! Number of ``uint16`` tokens.
    std::size_t num_tokens() const noexcept;

    //! Contiguous token array of length ``num_tokens()``.
    std::uint16_t const* data() const noexcept;

    std::uint16_t token_u16(std::size_t i) const;

    std::string const& path() const noexcept;

private:
    std::string path_;
    void* addr_ = nullptr;
    std::size_t size_ = 0;
    int fd_ = -1;
};

//! Yields non-overlapping training batches aligned to ``(seq_len+1)``-token
//! sequences. Truncates trailing tokens (same as the Python example).
class CausalLmBatchIterator
{
public:
    //! ``vocab_size``: if ``> 0``, ``next`` throws if any token is out of
    //! range.
    CausalLmBatchIterator(
        TokenMemoryMap const& tokens,
        CausalLmBatchConfig const& cfg,
        Index vocab_size);

    //! Fills ``batch``; sizes are ``n_seq * n_batch``. Returns ``false`` when
    //! no full batch remains.
    bool next(CausalLmBatch& batch);

    std::size_t num_batches() const noexcept;
    std::size_t batch_index() const noexcept;

private:
    TokenMemoryMap const* tokens_;
    CausalLmBatchConfig cfg_;
    Index vocab_size_;
    std::size_t num_seq_ = 0;
    std::size_t num_batches_ = 0;
    std::size_t batch_idx_ = 0;
    std::vector<std::size_t> seq_order_;
};

} // namespace nntile::graph::dataset
