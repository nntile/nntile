/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/kv_cache.hh
 * KVCache - reusable key-value cache for autoregressive inference.
 *
 * Manages per-layer K,V buffers and integrates with graph-based models.
 * Use with Llama, GPT, and other decoder-only models.
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/nn/graph_decl.hh>
#include <nntile/graph/tensor/graph_decl.hh>

namespace nntile::graph
{

//! KVCache - per-layer key-value cache for autoregressive generation
class KVCache
{
public:
    //! Per-layer cache shape: (head_size, max_seq, batch, n_head_kv)
    //! Layout matches Llama/GPT attention: head_size, seq, batch, heads
    struct Config
    {
        Index num_layers = 0;
        Index head_size = 0;
        Index n_head_kv = 0;
        Index max_seq = 0;
        Index batch = 1;
        DataType dtype = DataType::FP32;
    };

    //! Construct from config
    explicit KVCache(const Config& config);

    //! Construct from individual parameters (Llama-style layout)
    KVCache(Index num_layers,
            Index head_size,
            Index n_head_kv,
            Index max_seq,
            Index batch = 1,
            DataType dtype = DataType::FP32);

    // ── Core operations ─────────────────────────────────────────────────

    //! Reset cache: set length to 0, optionally zero buffers
    void reset(bool zero_buffers = false);

    //! Current valid length in cache (number of cached positions)
    Index len() const { return cache_len_; }

    //! Get cache for model.forward(): returns (tensor_pairs, cache_len).
    //! Call create_tensors() first to populate tensors_.
    const std::vector<std::pair<NNGraph::TensorNode*, NNGraph::TensorNode*>>*
    get_cache() const
    {
        return tensors_.empty() ? nullptr : &tensors_;
    }

    //! Advance cache length after a forward step (append was done by model)
    void advance(Index seq_len) { cache_len_ += seq_len; }

    //! Append raw K,V data to cache (for custom flows without graph).
    //! k_data, v_data: shape (head_size, seq_len, batch, n_head_kv), row-major.
    void append(const float* k_data,
                const float* v_data,
                Index seq_len);

    // ── Graph integration ─────────────────────────────────────────────

    //! Create tensor nodes in graph and return pairs for model.forward().
    //! Tensors are marked input+output for bind/update_from.
    std::vector<std::pair<NNGraph::TensorNode*, NNGraph::TensorNode*>>
    create_tensors(NNGraph* graph, const std::string& prefix = "kv_cache");

    //! Bind cache buffers to runtime (call before execute)
    void bind(TensorGraph::Runtime& runtime,
             const std::string& prefix = "kv_cache") const;

    //! Read updated cache from runtime (call after execute)
    void update_from(TensorGraph::Runtime& runtime,
                    const std::string& prefix = "kv_cache");

    // ── Buffer access (for custom flows) ─────────────────────────────────

    //! Get K buffer for layer (read/write)
    std::vector<float>& k_buffer(Index layer);
    const std::vector<float>& k_buffer(Index layer) const;

    //! Get V buffer for layer (read/write)
    std::vector<float>& v_buffer(Index layer);
    const std::vector<float>& v_buffer(Index layer) const;

    //! Number of layers
    Index num_layers() const { return config_.num_layers; }

    //! Cache shape per layer
    std::vector<Index> cache_shape() const;

private:
    Config config_;
    Index cache_len_ = 0;
    std::vector<std::vector<float>> k_buffers_;
    std::vector<std::vector<float>> v_buffers_;
    std::vector<std::pair<NNGraph::TensorNode*, NNGraph::TensorNode*>>
        tensors_;
};

} // namespace nntile::graph
