/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/tensor/tiling_config_json.cc
 * Tests for examples/tiling_config_json.hh
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <nntile/model/gpt2/gpt2_config.hh>
#include <nntile/tensor.hh>

#include "gpt2_axis_naming.hh"
#include "tiling_config_json.hh"

#include <cstdio>
#include <fstream>

using namespace nntile;
using namespace nntile::examples;

TEST_CASE("load tiling.json default + layers", "[tiling][json]")
{
    nlohmann::json j = {
        {"default", {{"hidden_size", 4}, {"intermediate_size", {2, 2}}}},
        {"layers",
            {{"h_1", {{"intermediate_size", {1, 1}}}}}}};

    FlatTilingSpec spec = load_tiling_from_json(j, 2);
    REQUIRE(spec.defaults.at("hidden_size").size() == 1);
    REQUIRE(spec.defaults.at("hidden_size")[0] == 4);
    REQUIRE(spec.per_layer.at(1).at("intermediate_size").size() == 2);
}

TEST_CASE("reject missing default or layers", "[tiling][json]")
{
    REQUIRE_THROWS(load_tiling_from_json(nlohmann::json::object(), 2));
    REQUIRE_THROWS(load_tiling_from_json(
        nlohmann::json{{"default", nlohmann::json::object()}}, 2));
}

TEST_CASE("parse tile sizes beyond int32 range", "[tiling][json]")
{
    Index const big = static_cast<Index>(3000000000LL);
    std::vector<Index> scalar =
        parse_tile_sizes_json(nlohmann::json(big), 0, "test");
    REQUIRE(scalar.size() == 1);
    REQUIRE(scalar[0] == big);

    std::vector<Index> array = parse_tile_sizes_json(
        nlohmann::json::array({big, big}), 0, "test");
    REQUIRE(array.size() == 2);
    REQUIRE(array[0] == big);
}

TEST_CASE("HF alias normalizes hidden_size", "[tiling][json]")
{
    nlohmann::json j = {
        {"default", {{"n_embd", 8}}},
        {"layers", nlohmann::json::object()}};
    FlatTilingSpec spec = load_tiling_from_json(j, 1);
    REQUIRE(spec.defaults.count("hidden_size") == 1);
}

TEST_CASE("apply per-layer override on named axes", "[tiling][json]")
{
    TensorGraph tg("apply");
    auto *t0 = tg.data({128})->set_name("model_transformer_h_0_mlp_w");
    auto *t1 = tg.data({128})->set_name("model_transformer_h_1_mlp_w");
    t0->axis(0)->name = "layer.0.intermediate_size";
    t1->axis(0)->name = "layer.1.intermediate_size";

    FlatTilingSpec spec;
    spec.defaults["intermediate_size"] = {64, 64};
    spec.per_layer[1]["intermediate_size"] = {40, 88};

    apply_flat_tiling_spec(tg, spec, 2);

    REQUIRE(t0->axis(0)->tile_sizes.size() == 2);
    REQUIRE(t0->axis(0)->tile_sizes[0] == 64);
    REQUIRE(t1->axis(0)->tile_sizes.size() == 2);
    REQUIRE(t1->axis(0)->tile_sizes[0] == 40);
}

TEST_CASE(
    "batch_size extent does not name attention head axes",
    "[tiling][naming]")
{
    model::gpt2::Gpt2Config cfg;
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.num_attention_heads = 4;
    cfg.num_hidden_layers = 1;
    cfg.validate();

    TensorGraph tg("naming");
    auto *heads =
        tg.data({4, 16, 64})->set_name("model_transformer_h_0_attn_q_weight");
    auto *batch_in = tg.data({8, 4})->set_name("input_ids");

    name_gpt2_training_axis_groups(tg, cfg, 8, 4);

    REQUIRE(heads->axis(0)->name == "layer.0.num_attention_heads");
    REQUIRE(batch_in->axis(1)->name == "batch_size");
}

TEST_CASE(
    "seq_len and batch_size named when extents match",
    "[tiling][naming]")
{
    model::gpt2::Gpt2Config cfg;
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.num_attention_heads = 4;
    cfg.num_hidden_layers = 1;
    cfg.validate();

    TensorGraph tg("naming");
    auto *batch_in = tg.data({8, 8})->set_name("input_ids");

    name_gpt2_training_axis_groups(tg, cfg, 8, 8);

    REQUIRE(batch_in->axis(0)->name == "seq_len");
    REQUIRE(batch_in->axis(1)->name == "batch_size");
}

TEST_CASE(
    "hidden_size not named seq_len when extents match",
    "[tiling][naming]")
{
    model::gpt2::Gpt2Config cfg;
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.num_attention_heads = 4;
    cfg.num_hidden_layers = 1;
    cfg.validate();

    TensorGraph tg("naming");
    auto *hidden = tg.data({64})->set_name("model_transformer_h_0_ln_1");

    name_gpt2_training_axis_groups(tg, cfg, 64, 4);

    REQUIRE(hidden->axis(0)->name == "hidden_size");
}

TEST_CASE("round-trip save and reload", "[tiling][json]")
{
    FlatTilingSpec spec;
    spec.defaults["batch_size"] = {1};
    spec.defaults["seq_len"] = {4, 4};
    spec.per_layer[0]["intermediate_size"] = {32, 32, 32, 32};

    std::string const path = "/tmp/nntile_test_tiling.json";
    save_tiling_json(spec, path);
    FlatTilingSpec loaded = load_tiling_json(path, 2);
    REQUIRE(loaded.defaults.at("batch_size")[0] == 1);
    REQUIRE(loaded.per_layer.at(0).at("intermediate_size")[0] == 32);
    std::remove(path.c_str());
}
