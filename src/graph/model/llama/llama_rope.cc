/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/llama/llama_rope.cc
 * RoPE sin/cos from position ids (HF default RoPE).
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/llama/llama_rope.hh"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nntile::model::llama
{

void rope_inv_freq_default(LlamaConfig const& config, float* out)
{
    config.validate();
    const Index half = config.head_dim / 2;
    if(out == nullptr || half <= 0 || (config.head_dim % 2) != 0)
    {
        throw std::invalid_argument(
            "rope_inv_freq_default: bad out or head_dim");
    }
    const double base = static_cast<double>(config.rope_theta);
    const double dim = static_cast<double>(config.head_dim);
    for(Index i = 0; i < half; ++i)
    {
        const double idx = static_cast<double>(2 * i);
        out[i] = static_cast<float>(1.0 / std::pow(base, idx / dim));
    }
}

void rope_sin_cos_from_position_ids(
    LlamaConfig const& config,
    std::int64_t const* position_ids,
    Index n_seq,
    Index n_batch,
    float* out_sin,
    float* out_cos)
{
    config.validate();
    if(position_ids == nullptr || out_sin == nullptr || out_cos == nullptr)
    {
        throw std::invalid_argument(
            "rope_sin_cos_from_position_ids: null pointer");
    }
    const Index half = config.head_dim / 2;
    if(half <= 0 || (config.head_dim % 2) != 0)
    {
        throw std::invalid_argument(
            "rope_sin_cos_from_position_ids: head_dim must be even");
    }
    std::vector<float> inv(static_cast<std::size_t>(half));
    rope_inv_freq_default(config, inv.data());
    for(Index b = 0; b < n_batch; ++b)
    {
        for(Index s = 0; s < n_seq; ++s)
        {
            const std::int64_t pos = position_ids[s + n_seq * b];
            for(Index h = 0; h < half; ++h)
            {
                const double angle =
                    static_cast<double>(pos) * static_cast<double>(inv[h]);
                const Index idx = h + half * (s + n_seq * b);
                out_cos[idx] = static_cast<float>(std::cos(angle));
                out_sin[idx] = static_cast<float>(std::sin(angle));
            }
        }
    }
}

} // namespace nntile::model::llama
