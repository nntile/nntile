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
#include "nntile/graph/dtype.hh"
#include "nntile/graph/nn/graph_data_node.hh"
#include "nntile/graph/tensor/graph_runtime.hh"

#include <cstring>
#include <stdexcept>

namespace nntile::graph
{

bool KVCache::uses_float_buffers_() const
{
    switch(config_.dtype)
    {
        case DataType::FP32:
        case DataType::FP32_FAST_TF32:
        case DataType::FP32_FAST_FP16:
        case DataType::FP32_FAST_BF16:
            return true;
        default:
            return false;
    }
}

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
    if(uses_float_buffers_())
    {
        k_buffers_f32_.resize(config_.num_layers);
        v_buffers_f32_.resize(config_.num_layers);
        for(Index i = 0; i < config_.num_layers; ++i)
        {
            k_buffers_f32_[i].resize(nelems, 0.0f);
            v_buffers_f32_[i].resize(nelems, 0.0f);
        }
    }
    else
    {
        size_t elem_size = dtype_size(config_.dtype);
        k_buffers_bytes_.resize(config_.num_layers);
        v_buffers_bytes_.resize(config_.num_layers);
        for(Index i = 0; i < config_.num_layers; ++i)
        {
            k_buffers_bytes_[i].resize(nelems * elem_size, std::byte{0});
            v_buffers_bytes_[i].resize(nelems * elem_size, std::byte{0});
        }
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
        if(uses_float_buffers_())
        {
            for(auto& buf : k_buffers_f32_)
                std::fill(buf.begin(), buf.end(), 0.0f);
            for(auto& buf : v_buffers_f32_)
                std::fill(buf.begin(), buf.end(), 0.0f);
        }
        else
        {
            for(auto& buf : k_buffers_bytes_)
                std::fill(buf.begin(), buf.end(), std::byte{0});
            for(auto& buf : v_buffers_bytes_)
                std::fill(buf.begin(), buf.end(), std::byte{0});
        }
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
    size_t nelems = static_cast<size_t>(config_.head_size * config_.max_seq
                                         * config_.batch * config_.n_head_kv);
    for(Index i = 0; i < config_.num_layers; ++i)
    {
        std::string k_name = prefix + "_k_" + std::to_string(i);
        std::string v_name = prefix + "_v_" + std::to_string(i);
        if(uses_float_buffers_())
        {
            runtime.bind_data(k_name, k_buffers_f32_[i]);
            runtime.bind_data(v_name, v_buffers_f32_[i]);
        }
        else
        {
            // Runtime expects float* for FP16/BF16 (it converts internally).
            // For FP64 it expects double*. Convert from our dtype-specific buffer.
            switch(config_.dtype)
            {
                case DataType::FP16:
                {
                    std::vector<float> k_tmp(nelems);
                    std::vector<float> v_tmp(nelems);
                    const nntile::fp16_t* k_src =
                        reinterpret_cast<const nntile::fp16_t*>(
                            k_buffers_bytes_[i].data());
                    const nntile::fp16_t* v_src =
                        reinterpret_cast<const nntile::fp16_t*>(
                            v_buffers_bytes_[i].data());
                    for(size_t j = 0; j < nelems; ++j)
                    {
                        k_tmp[j] = static_cast<float>(k_src[j]);
                        v_tmp[j] = static_cast<float>(v_src[j]);
                    }
                    runtime.bind_data(k_name, k_tmp);
                    runtime.bind_data(v_name, v_tmp);
                    break;
                }
                case DataType::BF16:
                {
                    std::vector<float> k_tmp(nelems);
                    std::vector<float> v_tmp(nelems);
                    const nntile::bf16_t* k_src =
                        reinterpret_cast<const nntile::bf16_t*>(
                            k_buffers_bytes_[i].data());
                    const nntile::bf16_t* v_src =
                        reinterpret_cast<const nntile::bf16_t*>(
                            v_buffers_bytes_[i].data());
                    for(size_t j = 0; j < nelems; ++j)
                    {
                        k_tmp[j] = static_cast<float>(k_src[j]);
                        v_tmp[j] = static_cast<float>(v_src[j]);
                    }
                    runtime.bind_data(k_name, k_tmp);
                    runtime.bind_data(v_name, v_tmp);
                    break;
                }
                case DataType::FP64:
                    runtime.bind_data(k_name,
                        reinterpret_cast<const double*>(
                            k_buffers_bytes_[i].data()), nelems);
                    runtime.bind_data(v_name,
                        reinterpret_cast<const double*>(
                            v_buffers_bytes_[i].data()), nelems);
                    break;
                default:
                    throw std::invalid_argument(
                        "KVCache::bind: unsupported dtype " +
                        dtype_to_string(config_.dtype));
            }
        }
    }
}

void KVCache::update_from(TensorGraph::Runtime& runtime,
                         const std::string& prefix)
{
    size_t nelems = static_cast<size_t>(config_.head_size * config_.max_seq
                                         * config_.batch * config_.n_head_kv);
    for(Index i = 0; i < config_.num_layers; ++i)
    {
        std::string k_name = prefix + "_k_" + std::to_string(i);
        std::string v_name = prefix + "_v_" + std::to_string(i);
        if(uses_float_buffers_())
        {
            k_buffers_f32_[i] = runtime.get_output<float>(k_name);
            v_buffers_f32_[i] = runtime.get_output<float>(v_name);
        }
        else
        {
            // Runtime returns float for FP16/BF16 (it converts from tensor).
            // For FP64 it returns double. Convert to our dtype-specific buffer.
            switch(config_.dtype)
            {
                case DataType::FP16:
                {
                    auto k_out = runtime.get_output<float>(k_name);
                    auto v_out = runtime.get_output<float>(v_name);
                    nntile::fp16_t* k_dst =
                        reinterpret_cast<nntile::fp16_t*>(
                            k_buffers_bytes_[i].data());
                    nntile::fp16_t* v_dst =
                        reinterpret_cast<nntile::fp16_t*>(
                            v_buffers_bytes_[i].data());
                    for(size_t j = 0; j < nelems; ++j)
                    {
                        k_dst[j] = nntile::fp16_t(k_out[j]);
                        v_dst[j] = nntile::fp16_t(v_out[j]);
                    }
                    break;
                }
                case DataType::BF16:
                {
                    auto k_out = runtime.get_output<float>(k_name);
                    auto v_out = runtime.get_output<float>(v_name);
                    nntile::bf16_t* k_dst =
                        reinterpret_cast<nntile::bf16_t*>(
                            k_buffers_bytes_[i].data());
                    nntile::bf16_t* v_dst =
                        reinterpret_cast<nntile::bf16_t*>(
                            v_buffers_bytes_[i].data());
                    for(size_t j = 0; j < nelems; ++j)
                    {
                        k_dst[j] = nntile::bf16_t(k_out[j]);
                        v_dst[j] = nntile::bf16_t(v_out[j]);
                    }
                    break;
                }
                case DataType::FP64:
                {
                    auto k_out = runtime.get_output<double>(k_name);
                    auto v_out = runtime.get_output<double>(v_name);
                    std::memcpy(k_buffers_bytes_[i].data(), k_out.data(),
                        k_out.size() * sizeof(double));
                    std::memcpy(v_buffers_bytes_[i].data(), v_out.data(),
                        v_out.size() * sizeof(double));
                    break;
                }
                default:
                    throw std::invalid_argument(
                        "KVCache::update_from: unsupported dtype " +
                        dtype_to_string(config_.dtype));
            }
        }
    }
}

std::vector<float>& KVCache::k_buffer(Index layer)
{
    if(layer < 0 || layer >= config_.num_layers)
        throw std::out_of_range("KVCache::k_buffer: layer out of range");
    if(!uses_float_buffers_())
    {
        throw std::runtime_error(
            "KVCache::k_buffer: buffer accessors only supported for "
            "FP32/FP32_FAST_* dtypes; use bind/update_from for " +
            dtype_to_string(config_.dtype));
    }
    return k_buffers_f32_[layer];
}

const std::vector<float>& KVCache::k_buffer(Index layer) const
{
    if(layer < 0 || layer >= config_.num_layers)
        throw std::out_of_range("KVCache::k_buffer: layer out of range");
    if(!uses_float_buffers_())
    {
        throw std::runtime_error(
            "KVCache::k_buffer: buffer accessors only supported for "
            "FP32/FP32_FAST_* dtypes; use bind/update_from for " +
            dtype_to_string(config_.dtype));
    }
    return k_buffers_f32_[layer];
}

std::vector<float>& KVCache::v_buffer(Index layer)
{
    if(layer < 0 || layer >= config_.num_layers)
        throw std::out_of_range("KVCache::v_buffer: layer out of range");
    if(!uses_float_buffers_())
    {
        throw std::runtime_error(
            "KVCache::v_buffer: buffer accessors only supported for "
            "FP32/FP32_FAST_* dtypes; use bind/update_from for " +
            dtype_to_string(config_.dtype));
    }
    return v_buffers_f32_[layer];
}

const std::vector<float>& KVCache::v_buffer(Index layer) const
{
    if(layer < 0 || layer >= config_.num_layers)
        throw std::out_of_range("KVCache::v_buffer: layer out of range");
    if(!uses_float_buffers_())
    {
        throw std::runtime_error(
            "KVCache::v_buffer: buffer accessors only supported for "
            "FP32/FP32_FAST_* dtypes; use bind/update_from for " +
            dtype_to_string(config_.dtype));
    }
    return v_buffers_f32_[layer];
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
    if(uses_float_buffers_())
    {
        for(Index layer = 0; layer < config_.num_layers; ++layer)
        {
            const float* k_src = k_data + layer * layer_size;
            const float* v_src = v_data + layer * layer_size;
            float* k_dst = k_buffers_f32_[layer].data();
            float* v_dst = v_buffers_f32_[layer].data();
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
    }
    else
    {
        for(Index layer = 0; layer < config_.num_layers; ++layer)
        {
            const float* k_src = k_data + layer * layer_size;
            const float* v_src = v_data + layer * layer_size;
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
                        switch(config_.dtype)
                        {
                            case DataType::FP16:
                            {
                                nntile::fp16_t* k_dst =
                                    reinterpret_cast<nntile::fp16_t*>(
                                        k_buffers_bytes_[layer].data());
                                nntile::fp16_t* v_dst =
                                    reinterpret_cast<nntile::fp16_t*>(
                                        v_buffers_bytes_[layer].data());
                                for(Index h = 0; h < H; ++h)
                                {
                                    k_dst[dst_offset + h] =
                                        nntile::fp16_t(k_src[src_offset + h]);
                                    v_dst[dst_offset + h] =
                                        nntile::fp16_t(v_src[src_offset + h]);
                                }
                                break;
                            }
                            case DataType::BF16:
                            {
                                nntile::bf16_t* k_dst =
                                    reinterpret_cast<nntile::bf16_t*>(
                                        k_buffers_bytes_[layer].data());
                                nntile::bf16_t* v_dst =
                                    reinterpret_cast<nntile::bf16_t*>(
                                        v_buffers_bytes_[layer].data());
                                for(Index h = 0; h < H; ++h)
                                {
                                    k_dst[dst_offset + h] =
                                        nntile::bf16_t(k_src[src_offset + h]);
                                    v_dst[dst_offset + h] =
                                        nntile::bf16_t(v_src[src_offset + h]);
                                }
                                break;
                            }
                            case DataType::FP64:
                            {
                                double* k_dst =
                                    reinterpret_cast<double*>(
                                        k_buffers_bytes_[layer].data());
                                double* v_dst =
                                    reinterpret_cast<double*>(
                                        v_buffers_bytes_[layer].data());
                                for(Index h = 0; h < H; ++h)
                                {
                                    k_dst[dst_offset + h] =
                                        static_cast<double>(k_src[src_offset + h]);
                                    v_dst[dst_offset + h] =
                                        static_cast<double>(v_src[src_offset + h]);
                                }
                                break;
                            }
                            default:
                                throw std::invalid_argument(
                                    "KVCache::append: unsupported dtype " +
                                    dtype_to_string(config_.dtype));
                        }
                    }
                }
            }
        }
    }
    cache_len_ += seq_len;
}

} // namespace nntile::graph
