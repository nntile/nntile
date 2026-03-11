/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/kv_cache.cc
 * KVCache implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/kv_cache.hh"
#include "nntile/graph/nn/graph_data_node.hh"
#include "nntile/graph/tensor/graph_runtime.hh"

#include <cstring>
#include <stdexcept>

namespace nntile::graph
{

KVCache::KVCache(const Config& config)
    : config_(config)
    , cache_len_(0)
{
    if(config_.num_layers <= 0 || config_.head_size <= 0
       || config_.n_head_kv <= 0 || config_.max_seq <= 0)
    {
        throw std::invalid_argument(
            "KVCache: invalid config (num_layers, head_size, n_head_kv, "
            "max_seq must be positive)");
    }
    size_t nelems = static_cast<size_t>(config_.head_size * config_.max_seq
                                        * config_.batch * config_.n_head_kv);
    k_buffers_.resize(config_.num_layers);
    v_buffers_.resize(config_.num_layers);
    for(Index i = 0; i < config_.num_layers; ++i)
    {
        k_buffers_[i].resize(nelems, 0.0f);
        v_buffers_[i].resize(nelems, 0.0f);
    }
}

KVCache::KVCache(Index num_layers,
                 Index head_size,
                 Index n_head_kv,
                 Index max_seq,
                 Index batch,
                 DataType dtype)
    : KVCache(Config{num_layers, head_size, n_head_kv, max_seq, batch, dtype})
{
}

void KVCache::reset(bool zero_buffers)
{
    cache_len_ = 0;
    tensors_.clear();
    if(zero_buffers)
    {
        for(auto& buf : k_buffers_)
            std::fill(buf.begin(), buf.end(), 0.0f);
        for(auto& buf : v_buffers_)
            std::fill(buf.begin(), buf.end(), 0.0f);
    }
}

std::vector<std::pair<NNGraph::TensorNode*, NNGraph::TensorNode*>>
KVCache::create_tensors(NNGraph* graph, const std::string& prefix)
{
    if(graph == nullptr)
        throw std::invalid_argument("KVCache::create_tensors: graph is null");

    tensors_.clear();
    tensors_.reserve(config_.num_layers);

    std::vector<Index> shape = {
        config_.head_size, config_.max_seq, config_.batch, config_.n_head_kv};

    for(Index i = 0; i < config_.num_layers; ++i)
    {
        std::string k_name = prefix + "_k_" + std::to_string(i);
        std::string v_name = prefix + "_v_" + std::to_string(i);
        auto* k_cache = graph->tensor(shape, k_name, config_.dtype, false);
        auto* v_cache = graph->tensor(shape, v_name, config_.dtype, false);
        k_cache->mark_input(true);
        k_cache->mark_output(true);
        v_cache->mark_input(true);
        v_cache->mark_output(true);
        tensors_.emplace_back(k_cache, v_cache);
    }

    return tensors_;
}

void KVCache::bind(TensorGraph::Runtime& runtime,
                   const std::string& prefix) const
{
    for(Index i = 0; i < config_.num_layers; ++i)
    {
        std::string k_name = prefix + "_k_" + std::to_string(i);
        std::string v_name = prefix + "_v_" + std::to_string(i);
        runtime.bind_data(k_name, k_buffers_[i]);
        runtime.bind_data(v_name, v_buffers_[i]);
    }
}

void KVCache::update_from(TensorGraph::Runtime& runtime,
                         const std::string& prefix)
{
    for(Index i = 0; i < config_.num_layers; ++i)
    {
        std::string k_name = prefix + "_k_" + std::to_string(i);
        std::string v_name = prefix + "_v_" + std::to_string(i);
        k_buffers_[i] = runtime.get_output<float>(k_name);
        v_buffers_[i] = runtime.get_output<float>(v_name);
    }
}

std::vector<float>& KVCache::k_buffer(Index layer)
{
    if(layer < 0 || layer >= config_.num_layers)
        throw std::out_of_range("KVCache::k_buffer: layer out of range");
    return k_buffers_[layer];
}

const std::vector<float>& KVCache::k_buffer(Index layer) const
{
    if(layer < 0 || layer >= config_.num_layers)
        throw std::out_of_range("KVCache::k_buffer: layer out of range");
    return k_buffers_[layer];
}

std::vector<float>& KVCache::v_buffer(Index layer)
{
    if(layer < 0 || layer >= config_.num_layers)
        throw std::out_of_range("KVCache::v_buffer: layer out of range");
    return v_buffers_[layer];
}

const std::vector<float>& KVCache::v_buffer(Index layer) const
{
    if(layer < 0 || layer >= config_.num_layers)
        throw std::out_of_range("KVCache::v_buffer: layer out of range");
    return v_buffers_[layer];
}

std::vector<Index> KVCache::cache_shape() const
{
    return {config_.head_size, config_.max_seq, config_.batch,
            config_.n_head_kv};
}

void KVCache::append(const float* k_data,
                     const float* v_data,
                     Index seq_len)
{
    if(seq_len <= 0)
        return;
    if(cache_len_ + seq_len > config_.max_seq)
    {
        throw std::invalid_argument(
            "KVCache::append: would exceed max_seq");
    }
    Index H = config_.head_size;
    Index S = config_.max_seq;
    Index B = config_.batch;
    Index N = config_.n_head_kv;
    size_t layer_size = static_cast<size_t>(H * seq_len * B * N);
    for(Index layer = 0; layer < config_.num_layers; ++layer)
    {
        const float* k_src = k_data + layer * layer_size;
        const float* v_src = v_data + layer * layer_size;
        float* k_dst = k_buffers_[layer].data();
        float* v_dst = v_buffers_[layer].data();
        for(Index s = 0; s < seq_len; ++s)
        {
            for(Index b = 0; b < B; ++b)
            {
                for(Index n = 0; n < N; ++n)
                {
                    size_t dst_offset = static_cast<size_t>(
                        (cache_len_ + s) * H + b * H * S + n * H * S * B);
                    size_t src_offset = static_cast<size_t>(
                        s * H + b * H * seq_len + n * H * seq_len * B);
                    std::memcpy(k_dst + dst_offset, k_src + src_offset,
                                static_cast<size_t>(H) * sizeof(float));
                    std::memcpy(v_dst + dst_offset, v_src + src_offset,
                                static_cast<size_t>(H) * sizeof(float));
                }
            }
        }
    }
    cache_len_ += seq_len;
}

} // namespace nntile::graph
