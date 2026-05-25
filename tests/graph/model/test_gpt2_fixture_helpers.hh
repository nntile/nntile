/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/test_gpt2_fixture_helpers.hh
 * Shared JSON, position_ids, and attention-mask helpers for GPT-2 graph model
 * tests.
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
#include <stdexcept>
#include <vector>

namespace nntile::test::gpt2_fixture
{

inline Index json_index(const nlohmann::json &o, const char *key)
{
    return static_cast<Index>(o.at(key).get<std::int64_t>());
}

inline bool load_position_ids(nntile::graph::NNGraph &g,
    const nntile::graph::io::SafeTensorsReader &reader,
    Index n_seq,
    Index n_batch,
    nntile::graph::NNGraph::TensorNode *&out_pos,
    std::vector<std::int64_t> &pos_data)
{
    out_pos = nullptr;
    pos_data.clear();
    if (!reader.has_tensor("position_ids"))
    {
        return false;
    }
    const auto &info = reader.tensor_info("position_ids");
    if (info.shape.size() != 2 || info.shape[0] != n_seq ||
        info.shape[1] != n_batch)
    {
        throw std::runtime_error(
            "GPT-2 test fixture: position_ids shape mismatch");
    }
    out_pos = g.tensor({n_seq, n_batch}, nntile::graph::DataType::INT64, false)
                  ->set_name("position_ids");
    auto raw = reader.read_tensor("position_ids");
    pos_data.resize(raw.size() / sizeof(std::int64_t));
    std::memcpy(pos_data.data(), raw.data(), raw.size());
    return true;
}

inline void mark_position_ids_input(
    nntile::graph::NNGraph::TensorNode *position_ids)
{
    if (position_ids != nullptr)
    {
        position_ids->mark_input(true);
    }
}

inline void bind_position_ids(nntile::graph::Runtime &runtime,
    nntile::graph::NNGraph::TensorNode *position_ids,
    const std::vector<std::int64_t> &pos_data)
{
    if (position_ids == nullptr)
    {
        return;
    }
    runtime.bind_data(position_ids, pos_data);
}

inline bool load_attn_mask_bool(nntile::graph::NNGraph &g,
    const nntile::graph::io::SafeTensorsReader &reader,
    Index n_seq,
    nntile::graph::NNGraph::TensorNode *&out_mask,
    std::vector<std::uint8_t> &mask_bytes)
{
    out_mask = nullptr;
    mask_bytes.clear();
    if (!reader.has_tensor("attn_mask"))
    {
        return false;
    }
    const auto &info = reader.tensor_info("attn_mask");
    if (info.shape.size() != 2 || info.shape[0] != n_seq ||
        info.shape[1] != n_seq)
    {
        throw std::runtime_error(
            "GPT-2 test fixture: attn_mask shape mismatch");
    }
    const auto n_el = static_cast<size_t>(n_seq * n_seq);
    out_mask = g.tensor({n_seq, n_seq}, nntile::graph::DataType::BOOL, false)
                   ->set_name("attn_mask");
    auto raw = reader.read_tensor("attn_mask");
    if (info.dtype == nntile::graph::DataType::BOOL)
    {
        if (raw.size() != n_el)
        {
            throw std::runtime_error(
                "GPT-2 test fixture: BOOL attn_mask byte size mismatch");
        }
        mask_bytes = std::move(raw);
        return true;
    }
    if (info.dtype == nntile::graph::DataType::FP32)
    {
        if (raw.size() != n_el * sizeof(float))
        {
            throw std::runtime_error(
                "GPT-2 test fixture: F32 attn_mask byte size mismatch");
        }
        mask_bytes.resize(n_el);
        const auto *p = reinterpret_cast<const float *>(raw.data());
        for (size_t i = 0; i < n_el; ++i)
        {
            mask_bytes[i] = (p[i] > 0.5f) ? static_cast<std::uint8_t>(1)
                                          : static_cast<std::uint8_t>(0);
        }
        return true;
    }
    throw std::runtime_error(
        "GPT-2 test fixture: attn_mask must be BOOL or F32");
}

inline void mark_mask_input(nntile::graph::NNGraph::TensorNode *mask)
{
    if (mask != nullptr)
    {
        mask->mark_input(true);
    }
}

inline void bind_mask_input(nntile::graph::Runtime &runtime,
    nntile::graph::NNGraph::TensorNode *mask,
    const std::vector<std::uint8_t> &mask_bytes)
{
    if (mask == nullptr)
    {
        return;
    }
    runtime.bind_data(mask, mask_bytes);
}

} // namespace nntile::test::gpt2_fixture
