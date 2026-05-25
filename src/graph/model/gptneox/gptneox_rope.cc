/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gptneox/gptneox_rope.cc
 * RoPE sin/cos from position ids (HF GPTNeoXRotaryEmbedding).
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gptneox/gptneox_rope.hh"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nntile::model::gptneox
{

Index gptneox_rope_dim(GptneoxConfig const& config)
{
    config.validate();
    double const pct = static_cast<double>(config.rotary_pct);
    Index dim = static_cast<Index>(
        std::lround(static_cast<double>(config.head_dim) * pct));
    if(dim < 2)
    {
        dim = 2;
    }
    if(dim % 2 != 0)
    {
        --dim;
    }
    if(dim > config.head_dim)
    {
        dim = config.head_dim;
        if(dim % 2 != 0)
        {
            --dim;
        }
    }
    return dim;
}

void rope_inv_freq_gptneox(GptneoxConfig const& config, float* out)
{
    config.validate();
    const Index rope_dim = gptneox_rope_dim(config);
    const Index half = rope_dim / 2;
    if(out == nullptr || half <= 0)
    {
        throw std::invalid_argument("rope_inv_freq_gptneox: bad out or rope_dim");
    }
    const double base = static_cast<double>(config.rotary_emb_base);
    const double dim = static_cast<double>(rope_dim);
    for(Index i = 0; i < half; ++i)
    {
        const double idx = static_cast<double>(2 * i);
        out[i] = static_cast<float>(1.0 / std::pow(base, idx / dim));
    }
}

void rope_sin_cos_from_position_ids(
    GptneoxConfig const& config,
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
    const Index rope_dim = gptneox_rope_dim(config);
    const Index half = rope_dim / 2;
    if(half <= 0)
    {
        throw std::invalid_argument(
            "rope_sin_cos_from_position_ids: invalid rope_dim");
    }
    std::vector<float> inv(static_cast<std::size_t>(half));
    rope_inv_freq_gptneox(config, inv.data());
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

} // namespace nntile::model::gptneox
