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

#include "tiling_config_json.hh"

#include <catch2/catch_test_macros.hpp>
#include <nntile/tensor.hh>

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
