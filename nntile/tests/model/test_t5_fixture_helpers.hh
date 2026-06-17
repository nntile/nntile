/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/test_t5_fixture_helpers.hh
 * Shared JSON and attention-mask helpers for T5 graph model tests.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nntile/graph.hh>
#include <nntile/io/safetensors.hh>
#include <nntile/model/t5/t5_config.hh>
#include <stdexcept>
#include <string>
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

//! Read and validate fixture JSON header (version 2, stem, safetensors name).
inline bool try_open_t5_fixture_json(const std::string &data_dir,
    const char *stem_cstr,
    std::string &stem_out,
    nlohmann::json &j_out)
{
    stem_out = stem_cstr;
    const std::string jpath = data_dir + "/" + stem_out + ".json";
    std::ifstream jf(jpath);
    if (!jf)
    {
        return false;
    }
    try
    {
        jf >> j_out;
        if (j_out.at("version").get<int>() != 2)
        {
            return false;
        }
        if (j_out.at("stem").get<std::string>() != stem_out)
        {
            return false;
        }
        const std::string expected_st = stem_out + ".safetensors";
        if (j_out.at("safetensors").get<std::string>() != expected_st)
        {
            return false;
        }
    }
    catch (...)
    {
        return false;
    }
    return true;
}

inline void load_t5_config_from_fixture_json(const nlohmann::json &j,
    model::t5::T5Config &config)
{
    const auto &T = j.at("t5");
    config.vocab_size = json_index(T, "vocab_size");
    config.d_model = json_index(T, "d_model");
    config.d_kv = json_index(T, "d_kv");
    config.d_ff = json_index(T, "d_ff");
    config.num_heads = json_index(T, "num_heads");
    config.num_layers = json_index(T, "num_layers");
    config.num_decoder_layers = json_index(T, "num_decoder_layers");
    config.layer_norm_epsilon = static_cast<float>(
        T.at("layer_norm_epsilon").get<double>());
    prepare_t5_config(config);
}

inline void load_t5_fixture_tolerances(const nlohmann::json &j,
    float &forward_tol,
    float &backward_tol)
{
    forward_tol = static_cast<float>(
        j.at("tolerances").at("forward").get<double>());
    backward_tol = static_cast<float>(
        j.at("tolerances").at("backward").get<double>());
}

inline std::string t5_fixture_safetensors_path(const std::string &data_dir,
    const std::string &stem)
{
    return data_dir + "/" + stem + ".safetensors";
}

inline bool load_attn_mask_bool(nntile::NNGraph &g,
    const nntile::io::SafeTensorsReader &reader,
    const char *tensor_name,
    Index n_k_seq,
    Index n_q_seq,
    nntile::NNGraph::TensorNode *&out_mask,
    std::vector<std::uint8_t> &mask_bytes)
{
    out_mask = nullptr;
    mask_bytes.clear();
    if (!reader.has_tensor(tensor_name))
    {
        return false;
    }
    const auto &info = reader.tensor_info(tensor_name);
    if (info.shape.size() != 2 || info.shape[0] != n_q_seq ||
        info.shape[1] != n_k_seq)
    {
        throw std::runtime_error(
            "T5 test fixture: attention mask shape mismatch");
    }
    const auto n_el = static_cast<size_t>(n_k_seq * n_q_seq);
    out_mask = g.tensor({n_q_seq, n_k_seq}, nntile::DataType::BOOL, false)
                   ->set_name(tensor_name);
    auto raw = reader.read_tensor(tensor_name);
    if (info.dtype == nntile::DataType::BOOL)
    {
        if (raw.size() != n_el)
        {
            throw std::runtime_error(
                "T5 test fixture: BOOL mask byte size mismatch");
        }
        mask_bytes = std::move(raw);
        return true;
    }
    if (info.dtype == nntile::DataType::FP32)
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

inline void mark_mask_input(nntile::NNGraph::TensorNode *mask)
{
    if (mask != nullptr)
    {
        mask->mark_input(true);
    }
}

inline void bind_mask_input(nntile::Runtime &runtime,
    nntile::NNGraph::TensorNode *mask,
    const std::vector<std::uint8_t> &mask_bytes)
{
    if (mask == nullptr)
    {
        return;
    }
    runtime.bind_data(mask, mask_bytes);
}

} // namespace nntile::test::t5_fixture
