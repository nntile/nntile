/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/test_llama_attention_fixture.hh
 * JSON + path helpers for Llama attention safetensors fixtures (shared by
 * ``llama_attention`` and ``llama_rope`` tests).
 *
 * @version 1.1.0
 * */

#pragma once

#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "test_llama_fixture_helpers.hh"
#include <nntile/graph/model/llama/llama_config.hh>

namespace nntile::test::llama_attention_fixture
{

//! Basenames (no extension) for paired ``.json`` / ``.safetensors`` in
//! ``LLAMA_DATA_DIR`` — must match ``generate_test_data.py`` output.
namespace attn_fixture_stem
{

constexpr char llama_attention[] = "llama_attention";
constexpr char llama_attention_no_rope[] = "llama_attention_no_rope";
constexpr char llama_attention_causal[] = "llama_attention_causal";
constexpr char llama_attention_no_rope_causal[] =
    "llama_attention_no_rope_causal";
constexpr char llama_attention_gqa[] = "llama_attention_gqa";
constexpr char llama_attention_gqa_no_rope[] =
    "llama_attention_gqa_no_rope";
constexpr char llama_attention_gqa_causal[] =
    "llama_attention_gqa_causal";
constexpr char llama_attention_gqa_no_rope_causal[] =
    "llama_attention_gqa_no_rope_causal";

} // namespace attn_fixture_stem

//! Parsed ``<stem>.json`` (``version`` 2) next to ``<stem>.safetensors``.
struct AttentionFixtureSpec
{
    nntile::model::llama::LlamaConfig config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_attention_fixture_spec(
    const std::string& data_dir,
    const char* stem_cstr,
    AttentionFixtureSpec& out)
{
    out = {};
    out.stem = stem_cstr;
    const std::string jpath = data_dir + "/" + out.stem + ".json";
    std::ifstream jf(jpath);
    if(!jf)
    {
        return false;
    }
    nlohmann::json j;
    try
    {
        jf >> j;
        if(j.at("version").get<int>() != 2)
        {
            return false;
        }
        if(j.at("stem").get<std::string>() != out.stem)
        {
            return false;
        }
        const std::string expected_st = out.stem + ".safetensors";
        if(j.at("safetensors").get<std::string>() != expected_st)
        {
            return false;
        }
        const auto& L = j.at("llama");
        out.config.hidden_size = llama_fixture::json_index(L, "hidden_size");
        out.config.num_attention_heads =
            llama_fixture::json_index(L, "num_attention_heads");
        out.config.num_key_value_heads =
            llama_fixture::json_index(L, "num_key_value_heads");
        out.config.compute_head_dim();
        out.hidden = out.config.hidden_size;
        out.seq = llama_fixture::json_index(j, "sequence_length");
        out.batch = llama_fixture::json_index(j, "batch");
        out.forward_tol = static_cast<float>(
            j.at("tolerances").at("forward").get<double>());
        out.backward_tol = static_cast<float>(
            j.at("tolerances").at("backward").get<double>());
    }
    catch(...)
    {
        return false;
    }
    return true;
}

inline std::string attention_fixture_safetensors_path(
    const std::string& data_dir,
    const AttentionFixtureSpec& spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

//! JSON + safetensors must both exist and JSON must parse.
inline bool skip_unless_fixture_ready(
    const char* stem,
    AttentionFixtureSpec& fx)
{
#ifdef LLAMA_DATA_DIR
    const std::string dir = std::string(LLAMA_DATA_DIR);
    if(!try_load_attention_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(attention_fixture_safetensors_path(dir, fx));
    return st.good();
#else
    (void)stem;
    (void)fx;
    return false;
#endif
}

} // namespace nntile::test::llama_attention_fixture
