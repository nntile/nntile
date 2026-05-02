/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/model/llama/llama_rope.cc
 * Tests for ``rope_sin_cos_from_position_ids`` vs HuggingFace fixtures
 * (``llama_attention`` safetensors from ``generate_test_data.py``).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "context_fixture.hh"
#include "test_llama_attention_fixture.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/llama/llama_config.hh"
#include "nntile/graph/model/llama/llama_rope.hh"

using namespace nntile;
using namespace nntile::graph::io;
using namespace nntile::model::llama;
using namespace nntile::test::llama_attention_fixture;

#ifndef LLAMA_DATA_DIR

TEST_CASE("Llama RoPE tests skipped (LLAMA_DATA_DIR undefined)", "[model][llama]")
{
    SKIP("LLAMA_DATA_DIR not defined at compile time.");
}

#else

static float max_abs_diff(
    const std::vector<float>& a,
    const std::vector<float>& b)
{
    REQUIRE(a.size() == b.size());
    float m = 0.f;
    for(std::size_t i = 0; i < a.size(); ++i)
    {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

//! ``rope_sin_cos_from_position_ids`` must match HF ``rope_sin`` / ``rope_cos``
//! in the attention fixture (requires ``position_ids`` in safetensors).
TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Llama RoPE sin/cos from position_ids matches HF fixture",
    "[model][llama][rope]")
{
    AttentionFixtureSpec fx;
    if(!skip_unless_fixture_ready(attn_fixture_stem::llama_attention, fx))
    {
        SKIP("Llama attention fixture pair not found.");
    }
    const std::string path =
        attention_fixture_safetensors_path(std::string(LLAMA_DATA_DIR), fx);
    SafeTensorsReader reader(path);
    if(!reader.has_tensor("position_ids"))
    {
        SKIP("Fixture missing position_ids; regenerate llama_data.");
    }
    const Index n_seq = fx.seq;
    const Index n_batch = fx.batch;
    std::vector<std::uint8_t> pos_bytes = reader.read_tensor("position_ids");
    const auto expected =
        static_cast<std::size_t>(n_seq * n_batch * sizeof(std::int64_t));
    REQUIRE(pos_bytes.size() == expected);
    std::vector<std::int64_t> pos(
        static_cast<std::size_t>(n_seq * n_batch));
    std::memcpy(pos.data(), pos_bytes.data(), pos_bytes.size());

    const Index half = fx.config.head_dim / 2;
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

    constexpr float k_tol = 2e-5f;
    REQUIRE(max_abs_diff(sin_comp, sin_ref) < k_tol);
    REQUIRE(max_abs_diff(cos_comp, cos_ref) < k_tol);
}

#endif
