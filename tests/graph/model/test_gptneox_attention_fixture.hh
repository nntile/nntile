/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/test_gptneox_attention_fixture.hh
 * JSON + path helpers for GPT-NeoX attention safetensors fixtures.
 *
 * @version 1.1.0
 * */

#pragma once

#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "test_gptneox_fixture_helpers.hh"
#include <nntile/graph/model/gptneox/gptneox_config.hh>

namespace nntile::test::gptneox_attention_fixture
{

//! Basenames (no extension) for paired ``.json`` / ``.safetensors`` in
//! ``GPTNEOX_DATA_DIR`` — must match ``generate_test_data.py`` output.
namespace attn_fixture_stem
{

constexpr char gptneox_attention[] = "gptneox_attention";
constexpr char gptneox_attention_no_rope[] = "gptneox_attention_no_rope";
constexpr char gptneox_attention_causal[] = "gptneox_attention_causal";
constexpr char gptneox_attention_no_rope_causal[] =
    "gptneox_attention_no_rope_causal";

} // namespace attn_fixture_stem

struct AttentionFixtureSpec
{
    nntile::model::gptneox::GptneoxConfig config{};
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
        const auto& G = j.at("gptneox");
        out.config.hidden_size = gptneox_fixture::json_index(G, "hidden_size");
        out.config.intermediate_size =
            gptneox_fixture::json_index(G, "intermediate_size");
        out.config.num_attention_heads =
            gptneox_fixture::json_index(G, "num_attention_heads");
        out.config.head_dim = gptneox_fixture::json_index(G, "head_dim");
        out.config.max_position_embeddings =
            gptneox_fixture::json_index(G, "max_position_embeddings");
        if(G.contains("rotary_pct"))
        {
            out.config.rotary_pct =
                static_cast<float>(G.at("rotary_pct").get<double>());
        }
        if(G.contains("rotary_emb_base"))
        {
            out.config.rotary_emb_base =
                static_cast<float>(G.at("rotary_emb_base").get<double>());
        }
        out.hidden = out.config.hidden_size;
        out.seq = gptneox_fixture::json_index(j, "sequence_length");
        out.batch = gptneox_fixture::json_index(j, "batch");
        out.forward_tol = static_cast<float>(
            j.at("tolerances").at("forward").get<double>());
        out.backward_tol = static_cast<float>(
            j.at("tolerances").at("backward").get<double>());
        out.config.validate();
    }
    catch(...)
    {
        return false;
    }
    gptneox_fixture::prepare_gptneox_config(out.config);
    return true;
}

inline std::string attention_fixture_safetensors_path(
    const std::string& data_dir,
    const AttentionFixtureSpec& spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

inline bool skip_unless_fixture_ready(
    const char* stem,
    AttentionFixtureSpec& fx)
{
#ifdef GPTNEOX_DATA_DIR
    const std::string dir = std::string(GPTNEOX_DATA_DIR);
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

} // namespace nntile::test::gptneox_attention_fixture
