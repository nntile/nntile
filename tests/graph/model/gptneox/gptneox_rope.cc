/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/gptneox/gptneox_rope.cc
 * Tests for ``rope_sin_cos_from_position_ids`` vs attention fixtures.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "context_fixture.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/gptneox/gptneox_config.hh"
#include "nntile/graph/model/gptneox/gptneox_rope.hh"
#include "test_gptneox_fixture_helpers.hh"

using namespace nntile;
using namespace nntile::graph::io;
using namespace nntile::model::gptneox;
using namespace nntile::test::gptneox_fixture;

#ifndef GPTNEOX_DATA_DIR

TEST_CASE(
    "Gptneox RoPE tests skipped (GPTNEOX_DATA_DIR undefined)", "[model][gptneox]")
{
    SKIP("GPTNEOX_DATA_DIR not defined at compile time.");
}

#else

namespace
{

struct AttentionFixtureSpec
{
    GptneoxConfig config{};
    Index seq = 0;
    Index batch = 0;
};

inline bool try_load_attention_fixture_spec(
    const std::string &data_dir,
    const char *stem_cstr,
    AttentionFixtureSpec &out)
{
    out = {};
    const std::string stem = stem_cstr;
    const std::string jpath = data_dir + "/" + stem + ".json";
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
        const auto &G = j.at("gptneox");
        out.config.hidden_size = json_index(G, "hidden_size");
        out.config.num_attention_heads = json_index(G, "num_attention_heads");
        out.config.head_dim = json_index(G, "head_dim");
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
        if(!G.contains("rotary_pct"))
        {
            out.config.rotary_pct = 1.0f;
        }
        out.seq = json_index(j, "sequence_length");
        out.batch = json_index(j, "batch");
        prepare_gptneox_config(out.config);
    }
    catch(...)
    {
        return false;
    }
    return true;
}

static float max_abs_diff(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    REQUIRE(a.size() == b.size());
    float m = 0.f;
    for(std::size_t i = 0; i < a.size(); ++i)
    {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Gptneox RoPE sin/cos from position_ids matches attention fixture",
    "[model][gptneox][rope]")
{
    AttentionFixtureSpec fx;
    if(!try_load_attention_fixture_spec(
           std::string(GPTNEOX_DATA_DIR), "gptneox_attention", fx))
    {
        SKIP("GPT-NeoX attention fixture not found.");
    }
    const std::string path =
        std::string(GPTNEOX_DATA_DIR) + "/gptneox_attention.safetensors";
    SafeTensorsReader reader(path);
    if(!reader.has_tensor("position_ids"))
    {
        SKIP("Fixture missing position_ids; regenerate gptneox_data.");
    }
    const Index n_seq = fx.seq;
    const Index n_batch = fx.batch;
    std::vector<std::uint8_t> pos_bytes = reader.read_tensor("position_ids");
    const auto expected =
        static_cast<std::size_t>(n_seq * n_batch * sizeof(std::int64_t));
    REQUIRE(pos_bytes.size() == expected);
    std::vector<std::int64_t> pos_c_flat(
        static_cast<std::size_t>(n_seq * n_batch));
    std::memcpy(pos_c_flat.data(), pos_bytes.data(), pos_bytes.size());
    // Safetensors flat layout is C-order; ``rope_sin_cos_from_position_ids``
    // expects ``pos[s + n_seq * b]`` (column-major ``(seq, batch)``).
    std::vector<std::int64_t> pos(
        static_cast<std::size_t>(n_seq * n_batch));
    for(Index b = 0; b < n_batch; ++b)
    {
        for(Index s = 0; s < n_seq; ++s)
        {
            pos[s + static_cast<std::size_t>(n_seq * b)] =
                pos_c_flat[s * static_cast<std::size_t>(n_batch) + b];
        }
    }

    const Index half = gptneox_rope_dim(fx.config) / 2;
    const std::size_t rope_elems =
        static_cast<std::size_t>(half * n_seq * n_batch);
    std::vector<float> sin_comp(rope_elems);
    std::vector<float> cos_comp(rope_elems);
    rope_sin_cos_from_position_ids(
        fx.config,
        pos.data(),
        n_seq,
        n_batch,
        sin_comp.data(),
        cos_comp.data());

    std::vector<std::uint8_t> sin_ref_b = reader.read_tensor("rope_sin");
    std::vector<std::uint8_t> cos_ref_b = reader.read_tensor("rope_cos");
    std::vector<float> sin_ref(sin_ref_b.size() / sizeof(float));
    std::vector<float> cos_ref(cos_ref_b.size() / sizeof(float));
    std::memcpy(sin_ref.data(), sin_ref_b.data(), sin_ref_b.size());
    std::memcpy(cos_ref.data(), cos_ref_b.data(), cos_ref_b.size());

    constexpr float k_tol = 1e-5f;
    REQUIRE(max_abs_diff(sin_comp, sin_ref) < k_tol);
    REQUIRE(max_abs_diff(cos_comp, cos_ref) < k_tol);
}

#endif
