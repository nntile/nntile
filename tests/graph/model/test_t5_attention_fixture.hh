/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/test_t5_attention_fixture.hh
 * JSON + path helpers for T5 self-attention safetensors fixtures.
 *
 * @version 1.1.0
 * */

#pragma once

#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "test_t5_fixture_helpers.hh"

namespace nntile::test::t5_attention_fixture
{

//! Basenames for paired ``.json`` / ``.safetensors`` in ``T5_DATA_DIR``.
//! T5 graph attention has no RoPE; stems are explicit for the mask matrix.
namespace attn_fixture_stem
{

constexpr char t5_attention[] = "t5_attention";
constexpr char t5_attention_causal[] = "t5_attention_causal";
constexpr char t5_attention_no_rope_nomask[] = "t5_attention_no_rope_nomask";
constexpr char t5_attention_no_rope_causal[] = "t5_attention_no_rope_causal";

} // namespace attn_fixture_stem

struct AttentionFixtureSpec
{
    nntile::model::t5::T5Config config{};
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
    nlohmann::json j;
    if(!t5_fixture::try_open_t5_fixture_json(data_dir, stem_cstr, out.stem, j))
    {
        return false;
    }
    try
    {
        t5_fixture::load_t5_config_from_fixture_json(j, out.config);
        out.hidden = out.config.d_model;
        out.seq = t5_fixture::json_index(j, "sequence_length");
        out.batch = t5_fixture::json_index(j, "batch");
        t5_fixture::load_t5_fixture_tolerances(
            j, out.forward_tol, out.backward_tol);
        t5_fixture::prepare_t5_config(out.config);
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
    return t5_fixture::t5_fixture_safetensors_path(data_dir, spec.stem);
}

inline bool skip_unless_fixture_ready(
    const char* stem,
    AttentionFixtureSpec& fx)
{
#ifdef T5_DATA_DIR
    const std::string dir = std::string(T5_DATA_DIR);
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

} // namespace nntile::test::t5_attention_fixture
