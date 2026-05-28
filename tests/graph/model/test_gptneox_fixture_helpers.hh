/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/test_gptneox_fixture_helpers.hh
 * Shared JSON, RoPE, and attention-mask helpers for GPT-NeoX graph model tests.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <nlohmann/json.hpp>
#include <nntile/graph.hh>
#include <nntile/graph/io/safetensors.hh>
#include <nntile/graph/model/gptneox/gptneox_config.hh>
#include <nntile/graph/model/gptneox/gptneox_rope.hh>
#include <stdexcept>
#include <vector>

#include "test_safetensors_nntile_layout.hh"

namespace nntile::graph::test::gptneox_fixture
{

inline void prepare_gptneox_config(
    nntile::graph::model::gptneox::GptneoxConfig &cfg)
{
    cfg.compute_head_dim();
    cfg.validate();
}

inline Index json_index(const nlohmann::json &o, const char *key)
{
    return static_cast<Index>(o.at(key).get<std::int64_t>());
}

struct GptneoxRopeInputs
{
    nntile::graph::NNGraph::TensorNode *sin = nullptr;
    nntile::graph::NNGraph::TensorNode *cos = nullptr;
    std::vector<float> sin_data;
    std::vector<float> cos_data;
};

inline bool load_gptneox_rope_inputs(
    nntile::graph::NNGraph &g,
    const nntile::graph::io::SafeTensorsReader &reader,
    const nntile::graph::model::gptneox::GptneoxConfig &config,
    Index n_seq,
    Index n_batch,
    GptneoxRopeInputs &out)
{
    out = {};
    if(!reader.has_tensor("rope_sin") || !reader.has_tensor("rope_cos"))
    {
        return false;
    }
    const Index rope_dim =
        nntile::graph::model::gptneox::gptneox_rope_dim(config);
    const Index half = rope_dim / 2;
    if(half <= 0)
    {
        return false;
    }
    out.sin = g.tensor({half, n_seq, n_batch}, nntile::graph::DataType::FP32)
                  ->set_name("rope_sin");
    out.cos = g.tensor({half, n_seq, n_batch}, nntile::graph::DataType::FP32)
                  ->set_name("rope_cos");
    safetensors_nntile_layout::read_tensor_nntile_fortran(
        reader, "rope_sin", out.sin_data);
    safetensors_nntile_layout::read_tensor_nntile_fortran(
        reader, "rope_cos", out.cos_data);
    return true;
}

inline void mark_rope_inputs(const GptneoxRopeInputs &rope)
{
    if(rope.sin == nullptr)
    {
        return;
    }
    rope.sin->mark_input(true);
    rope.cos->mark_input(true);
}

inline void bind_rope_inputs(
    nntile::graph::Runtime &runtime, const GptneoxRopeInputs &rope)
{
    if(rope.sin == nullptr)
    {
        return;
    }
    runtime.bind_data(rope.sin, rope.sin_data);
    runtime.bind_data(rope.cos, rope.cos_data);
}

inline bool load_attn_mask_bool(nntile::graph::NNGraph &g,
    const nntile::graph::io::SafeTensorsReader &reader,
    Index n_seq,
    nntile::graph::NNGraph::TensorNode *&out_mask,
    std::vector<std::uint8_t> &mask_bytes)
{
    out_mask = nullptr;
    mask_bytes.clear();
    if(!reader.has_tensor("attn_mask"))
    {
        return false;
    }
    const auto &info = reader.tensor_info("attn_mask");
    const auto n_el = static_cast<size_t>(n_seq * n_seq);
    if(info.shape.size() == 1)
    {
        if(info.shape[0] != static_cast<Index>(n_el))
        {
            throw std::runtime_error(
                "GPT-NeoX test fixture: 1D attn_mask length mismatch");
        }
    }
    else if(
        info.shape.size() != 2 || info.shape[0] != n_seq ||
        info.shape[1] != n_seq)
    {
        throw std::runtime_error(
            "GPT-NeoX test fixture: attn_mask shape mismatch");
    }
    out_mask = g.tensor({n_seq, n_seq}, nntile::graph::DataType::BOOL, false)
                   ->set_name("attn_mask");
    auto raw = reader.read_tensor("attn_mask");
    if(info.dtype == nntile::graph::DataType::BOOL)
    {
        if(raw.size() != n_el)
        {
            throw std::runtime_error(
                "GPT-NeoX test fixture: BOOL attn_mask byte size mismatch");
        }
        mask_bytes = std::move(raw);
        return true;
    }
    if(info.dtype == nntile::graph::DataType::FP32)
    {
        if(raw.size() != n_el * sizeof(float))
        {
            throw std::runtime_error(
                "GPT-NeoX test fixture: F32 attn_mask byte size mismatch");
        }
        mask_bytes.resize(n_el);
        const auto *p = reinterpret_cast<const float *>(raw.data());
        for(size_t i = 0; i < n_el; ++i)
        {
            mask_bytes[i] = (p[i] > 0.5f) ? static_cast<std::uint8_t>(1)
                                          : static_cast<std::uint8_t>(0);
        }
        return true;
    }
    throw std::runtime_error(
        "GPT-NeoX test fixture: attn_mask must be BOOL or F32");
}

inline void mark_mask_input(nntile::graph::NNGraph::TensorNode *mask)
{
    if(mask != nullptr)
    {
        mask->mark_input(true);
    }
}

inline void bind_mask_input(nntile::graph::Runtime &runtime,
    nntile::graph::NNGraph::TensorNode *mask,
    const std::vector<std::uint8_t> &mask_bytes)
{
    if(mask == nullptr)
    {
        return;
    }
    runtime.bind_data(mask, mask_bytes);
}

inline void fill_sdpa_causal_mask_bytes(
    Index n_seq, std::vector<std::uint8_t> &mask_bytes)
{
    mask_bytes.assign(static_cast<size_t>(n_seq * n_seq), 0);
    for(Index query = 0; query < n_seq; ++query)
    {
        for(Index key = 0; key < n_seq; ++key)
        {
            mask_bytes[static_cast<size_t>(key + query * n_seq)] =
                (key <= query) ? static_cast<std::uint8_t>(1)
                               : static_cast<std::uint8_t>(0);
        }
    }
}

} // namespace nntile::graph::test::gptneox_fixture
