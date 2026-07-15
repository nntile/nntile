/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/test_roberta_fixture_helpers.hh
 * Shared helpers for RoBERTa graph model tests.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <nlohmann/json.hpp>
#include <nntile/graph.hh>
#include <nntile/io/safetensors.hh>
#include <stdexcept>
#include <vector>

namespace nntile::test::roberta_fixture
{

inline Index json_index(const nlohmann::json &o, const char *key)
{
    return static_cast<Index>(o.at(key).get<std::int64_t>());
}

inline bool load_position_ids(nntile::NNGraph &g,
    const nntile::io::SafeTensorsReader &reader,
    Index n_seq,
    Index n_batch,
    nntile::NNGraph::TensorNode *&out_pos,
    std::vector<std::int64_t> &pos_data)
{
    out_pos = nullptr;
    pos_data.clear();
    if (!reader.has_tensor("position_ids"))
    {
        return false;
    }
    const auto &info = reader.tensor_info("position_ids");
    if (info.shape.size() != 2 || info.shape[0] != n_batch ||
        info.shape[1] != n_seq)
    {
        throw std::runtime_error(
            "RoBERTa test fixture: position_ids shape mismatch");
    }
    out_pos = g.tensor({n_batch, n_seq}, nntile::DataType::INT64, false)
                  ->set_name("position_ids");
    auto raw = reader.read_tensor("position_ids");
    pos_data.resize(raw.size() / sizeof(std::int64_t));
    std::memcpy(pos_data.data(), raw.data(), raw.size());
    return true;
}

inline void mark_position_input(
    nntile::NNGraph::TensorNode *position_ids)
{
}

inline void bind_position_input(nntile::Runtime &runtime,
    nntile::NNGraph::TensorNode *position_ids,
    const std::vector<std::int64_t> &pos_data)
{
    if (position_ids != nullptr)
    {
        runtime.bind_data(position_ids, pos_data);
    }
}

inline bool load_attn_mask_bool(nntile::NNGraph &g,
    const nntile::io::SafeTensorsReader &reader,
    Index n_seq,
    nntile::NNGraph::TensorNode *&out_mask,
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
            "RoBERTa test fixture: attn_mask shape mismatch");
    }
    if (info.dtype != nntile::DataType::BOOL)
    {
        return false;
    }
    out_mask = g.tensor({n_seq, n_seq}, nntile::DataType::BOOL, false)
                   ->set_name("attn_mask");
    auto raw = reader.read_tensor("attn_mask");
    mask_bytes = reader.read_tensor("attn_mask");
    return true;
}

inline void mark_mask_input(nntile::NNGraph::TensorNode *mask)
{
}

inline void bind_mask_input(nntile::Runtime &runtime,
    nntile::NNGraph::TensorNode *mask,
    const std::vector<std::uint8_t> &mask_bytes)
{
    if (mask != nullptr)
    {
        runtime.bind_data(mask, mask_bytes);
    }
}

} // namespace nntile::test::roberta_fixture
