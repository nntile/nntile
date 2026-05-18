/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/test_llama_fixture_helpers.hh
 * Shared JSON, RoPE, and optional attention-mask helpers for Llama graph model
 * tests (``llama_attention``, ``llama_causal``, ``llama_decoder``).
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include <nntile/graph.hh>
#include <nntile/graph/io/safetensors.hh>
#include <nntile/graph/model/llama/llama_config.hh>

namespace nntile::test::llama_fixture
{

//! Read a JSON integer field as ``Index`` (fixture ``*.json`` schema).
inline Index json_index(const nlohmann::json& o, const char* key)
{
    return static_cast<Index>(o.at(key).get<std::int64_t>());
}

//! Optional RoPE tensors from safetensors (same layout as Python LlamaAttention).
struct LlamaRopeInputs
{
    nntile::graph::NNGraph::TensorNode* sin = nullptr;
    nntile::graph::NNGraph::TensorNode* cos = nullptr;
    std::vector<float> sin_data;
    std::vector<float> cos_data;
};

inline bool load_llama_rope_inputs(
    nntile::graph::NNGraph& g,
    const nntile::graph::io::SafeTensorsReader& reader,
    const nntile::model::llama::LlamaConfig& config,
    Index n_seq,
    Index n_batch,
    LlamaRopeInputs& out)
{
    out = {};
    if(!reader.has_tensor("rope_sin") || !reader.has_tensor("rope_cos"))
    {
        return false;
    }
    const Index head_dim = config.head_dim;
    if(head_dim % 2 != 0)
    {
        return false;
    }
    const Index half = head_dim / 2;
    out.sin =
        g.tensor({half, n_seq, n_batch}, "rope_sin", nntile::graph::DataType::FP32);
    out.cos =
        g.tensor({half, n_seq, n_batch}, "rope_cos", nntile::graph::DataType::FP32);
    auto read_f = [&](const char* name, std::vector<float>& dst) {
        std::vector<std::uint8_t> b = reader.read_tensor(name);
        dst.resize(b.size() / sizeof(float));
        std::memcpy(dst.data(), b.data(), b.size());
    };
    read_f("rope_sin", out.sin_data);
    read_f("rope_cos", out.cos_data);
    return true;
}

inline void mark_rope_inputs(const LlamaRopeInputs& rope)
{
    if(rope.sin == nullptr)
    {
        return;
    }
    rope.sin->mark_input(true);
    rope.cos->mark_input(true);
}

inline void bind_rope_inputs(
    nntile::graph::TileGraph::Runtime& runtime, const LlamaRopeInputs& rope)
{
    if(rope.sin == nullptr)
    {
        return;
    }
    runtime.bind_data("rope_sin", rope.sin_data);
    runtime.bind_data("rope_cos", rope.cos_data);
}

//! Optional causal mask ``(seq, seq)`` (1 = keep logit).
inline bool load_attn_mask_bool(
    nntile::graph::NNGraph& g,
    const nntile::graph::io::SafeTensorsReader& reader,
    Index n_seq,
    nntile::graph::NNGraph::TensorNode*& out_mask,
    std::vector<std::uint8_t>& mask_bytes)
{
    out_mask = nullptr;
    mask_bytes.clear();
    if(!reader.has_tensor("attn_mask"))
    {
        return false;
    }
    const auto& info = reader.tensor_info("attn_mask");
    if(info.shape.size() != 2 || info.shape[0] != n_seq
        || info.shape[1] != n_seq)
    {
        throw std::runtime_error(
            "Llama test fixture: attn_mask shape mismatch");
    }
    const auto n_el = static_cast<size_t>(n_seq * n_seq);
    out_mask = g.tensor(
        {n_seq, n_seq}, "attn_mask", nntile::graph::DataType::BOOL, false);
    auto raw = reader.read_tensor("attn_mask");
    if(info.dtype == nntile::graph::DataType::BOOL)
    {
        if(raw.size() != n_el)
        {
            throw std::runtime_error(
                "Llama test fixture: BOOL attn_mask byte size mismatch");
        }
        mask_bytes = std::move(raw);
        return true;
    }
    if(info.dtype == nntile::graph::DataType::FP32)
    {
        if(raw.size() != n_el * sizeof(float))
        {
            throw std::runtime_error(
                "Llama test fixture: F32 attn_mask byte size mismatch");
        }
        mask_bytes.resize(n_el);
        const auto* p = reinterpret_cast<const float*>(raw.data());
        for(size_t i = 0; i < n_el; ++i)
        {
            mask_bytes[i] =
                (p[i] > 0.5f) ? static_cast<std::uint8_t>(1)
                              : static_cast<std::uint8_t>(0);
        }
        return true;
    }
    throw std::runtime_error(
        "Llama test fixture: attn_mask must be BOOL or F32");
}

inline void mark_mask_input(nntile::graph::NNGraph::TensorNode* mask)
{
    if(mask != nullptr)
    {
        mask->mark_input(true);
    }
}

inline void bind_mask_input(
    nntile::graph::TileGraph::Runtime& runtime,
    nntile::graph::NNGraph::TensorNode* mask,
    const std::vector<std::uint8_t>& mask_bytes)
{
    if(mask == nullptr)
    {
        return;
    }
    runtime.bind_data(mask->name(), mask_bytes);
}

} // namespace nntile::test::llama_fixture
