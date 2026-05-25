/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/test_t5_fixture_helpers.hh
 * Shared JSON and attention-mask helpers for T5 graph model tests.
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

namespace nntile::test::t5_fixture
{

inline Index json_index(const nlohmann::json &o, const char *key)
{
    return static_cast<Index>(o.at(key).get<std::int64_t>());
}

inline void prepare_t5_config(model::t5::T5Config &config)
{
    config.validate();
}

inline bool load_attn_mask_bool(nntile::graph::NNGraph &g,
    const nntile::graph::io::SafeTensorsReader &reader,
    const char *tensor_name,
    Index n_k_seq,
    Index n_q_seq,
    nntile::graph::NNGraph::TensorNode *&out_mask,
    std::vector<std::uint8_t> &mask_bytes)
{
    out_mask = nullptr;
    mask_bytes.clear();
    if (!reader.has_tensor(tensor_name))
    {
        return false;
    }
    const auto &info = reader.tensor_info(tensor_name);
    if (info.shape.size() != 2 || info.shape[0] != n_k_seq ||
        info.shape[1] != n_q_seq)
    {
        throw std::runtime_error(
            "T5 test fixture: attention mask shape mismatch");
    }
    const auto n_el = static_cast<size_t>(n_k_seq * n_q_seq);
    out_mask = g.tensor({n_k_seq, n_q_seq}, nntile::graph::DataType::BOOL, false)
                   ->set_name(tensor_name);
    auto raw = reader.read_tensor(tensor_name);
    if (info.dtype == nntile::graph::DataType::BOOL)
    {
        if (raw.size() != n_el)
        {
            throw std::runtime_error(
                "T5 test fixture: BOOL mask byte size mismatch");
        }
        mask_bytes = std::move(raw);
        return true;
    }
    if (info.dtype == nntile::graph::DataType::FP32)
    {
        if (raw.size() != n_el * sizeof(float))
        {
            throw std::runtime_error(
                "T5 test fixture: F32 mask byte size mismatch");
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
        "T5 test fixture: attention mask must be BOOL or F32");
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

} // namespace nntile::test::t5_fixture
