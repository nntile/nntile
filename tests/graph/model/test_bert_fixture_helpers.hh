/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/test_bert_fixture_helpers.hh
 * Shared helpers for BERT graph model tests.
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

namespace nntile::graph::test::bert_fixture
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
            "BERT test fixture: position_ids shape mismatch");
    }
    out_pos = g.tensor({n_seq, n_batch}, nntile::graph::DataType::INT64, false)
                  ->set_name("position_ids");
    auto raw = reader.read_tensor("position_ids");
    pos_data.resize(raw.size() / sizeof(std::int64_t));
    std::memcpy(pos_data.data(), raw.data(), raw.size());
    return true;
}

inline bool load_token_type_ids(nntile::graph::NNGraph &g,
    const nntile::graph::io::SafeTensorsReader &reader,
    Index n_seq,
    Index n_batch,
    nntile::graph::NNGraph::TensorNode *&out_tt,
    std::vector<std::int64_t> &tt_data)
{
    out_tt = nullptr;
    tt_data.clear();
    if (!reader.has_tensor("token_type_ids"))
    {
        return false;
    }
    const auto &info = reader.tensor_info("token_type_ids");
    if (info.shape.size() != 2 || info.shape[0] != n_seq ||
        info.shape[1] != n_batch)
    {
        throw std::runtime_error(
            "BERT test fixture: token_type_ids shape mismatch");
    }
    out_tt = g.tensor({n_seq, n_batch}, nntile::graph::DataType::INT64, false)
                 ->set_name("token_type_ids");
    auto raw = reader.read_tensor("token_type_ids");
    tt_data.resize(raw.size() / sizeof(std::int64_t));
    std::memcpy(tt_data.data(), raw.data(), raw.size());
    return true;
}

inline void mark_ids_inputs(
    nntile::graph::NNGraph::TensorNode *position_ids,
    nntile::graph::NNGraph::TensorNode *token_type_ids)
{
    if (position_ids != nullptr)
    {
        position_ids->mark_input(true);
    }
    if (token_type_ids != nullptr)
    {
        token_type_ids->mark_input(true);
    }
}

inline void bind_ids_inputs(nntile::graph::Runtime &runtime,
    nntile::graph::NNGraph::TensorNode *position_ids,
    const std::vector<std::int64_t> &pos_data,
    nntile::graph::NNGraph::TensorNode *token_type_ids,
    const std::vector<std::int64_t> &tt_data)
{
    if (position_ids != nullptr)
    {
        runtime.bind_data(position_ids, pos_data);
    }
    if (token_type_ids != nullptr)
    {
        runtime.bind_data(token_type_ids, tt_data);
    }
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
            "BERT test fixture: attn_mask shape mismatch");
    }
    if (info.dtype != nntile::graph::DataType::BOOL)
    {
        return false;
    }
    out_mask = g.tensor({n_seq, n_seq}, nntile::graph::DataType::BOOL, false)
                   ->set_name("attn_mask");
    mask_bytes = reader.read_tensor("attn_mask");
    return true;
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
    if (mask != nullptr)
    {
        runtime.bind_data(mask, mask_bytes);
    }
}

} // namespace nntile::graph::test::bert_fixture
