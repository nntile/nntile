/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/t5/t5_ff.cc
 * Tests for T5LayerFF.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/t5/t5_ff.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/t5/t5_config.hh"
#include "test_frobenius.hh"
#include "test_t5_fixture_helpers.hh"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace nntile::core;
using namespace nntile::graph;
using namespace nntile::graph::model::t5;
using namespace nntile::graph::io;

#ifndef T5_DATA_DIR

TEST_CASE(
    "T5LayerFF tests skipped (T5_DATA_DIR undefined)", "[model][t5]")
{
    SKIP("T5_DATA_DIR not defined at compile time.");
}

#else

namespace ff_fixture_stem
{

constexpr char t5_ff[] = "t5_ff";

} // namespace ff_fixture_stem

namespace
{

using namespace nntile::graph::test::t5_fixture;

struct FfFixtureSpec
{
    T5Config config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_ff_fixture_spec(
    const std::string &data_dir, const char *stem_cstr, FfFixtureSpec &out)
{
    out = {};
    nlohmann::json j;
    if (!try_open_t5_fixture_json(data_dir, stem_cstr, out.stem, j))
    {
        return false;
    }
    try
    {
        load_t5_config_from_fixture_json(j, out.config);
        out.hidden = out.config.d_model;
        out.seq = json_index(j, "sequence_length");
        out.batch = json_index(j, "batch");
        load_t5_fixture_tolerances(j, out.forward_tol, out.backward_tol);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

inline bool skip_unless_fixture_ready(const char *stem, FfFixtureSpec &fx)
{
    const std::string dir = std::string(T5_DATA_DIR);
    if (!try_load_ff_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(t5_fixture_safetensors_path(dir, fx.stem));
    return st.good();
}

} // namespace

TEST_CASE("T5LayerFF forward builds output", "[model][t5]")
{
    FfFixtureSpec fx;
    if (!skip_unless_fixture_ready(ff_fixture_stem::t5_ff, fx))
    {
        SKIP("Missing or invalid t5_ff.json / .safetensors.");
    }
    NNGraph g("t5_ff");
    T5LayerFF ff(&g, "ff", fx.config);
    auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    auto *output = ff.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
    REQUIRE(ff.parameters_recursive().size() == 4);
}

TEST_CASE("T5LayerFF load from safetensors roundtrip", "[model][t5][io]")
{
    FfFixtureSpec fx;
    if (!skip_unless_fixture_ready(ff_fixture_stem::t5_ff, fx))
    {
        SKIP("Missing or invalid t5_ff.json / .safetensors.");
    }
    const std::string data_path =
        t5_fixture_safetensors_path(std::string(T5_DATA_DIR), fx.stem);

    NNGraph g1("load_graph");
    T5LayerFF ff1(&g1, "ff", fx.config);
    ff1.load(data_path);

    const std::string save_path = "/tmp/nntile_t5_ff_roundtrip.safetensors";
    ff1.save(save_path);

    SafeTensorsReader reader(data_path);
    SafeTensorsReader reader2(save_path);
    for (const auto &name : reader2.tensor_names())
    {
        REQUIRE(reader.has_tensor(name));
        REQUIRE(reader.read_tensor(name) == reader2.read_tensor(name));
    }
    std::remove(save_path.c_str());
}

TEST_CASE_METHOD(nntile::core::test::ContextFixture,
    "T5LayerFF matches PyTorch reference",
    "[model][t5]")
{
    FfFixtureSpec fx;
    if (!skip_unless_fixture_ready(ff_fixture_stem::t5_ff, fx))
    {
        SKIP("T5 FF fixture pair not found.");
    }
    const std::string full_path =
        t5_fixture_safetensors_path(std::string(T5_DATA_DIR), fx.stem);

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("ff_ref");
        auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                          ->set_name("input");
        T5LayerFF ff(&g, "ff", fx.config);
        ff.load(full_path);
        auto *output = ff.forward(input);
        input->mark_input(true);
        output->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input, input_data);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output);
    }

    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, fx.forward_tol);
}

TEST_CASE_METHOD(nntile::core::test::ContextFixture,
    "T5LayerFF backward matches PyTorch reference",
    "[model][t5]")
{
    FfFixtureSpec fx;
    if (!skip_unless_fixture_ready(ff_fixture_stem::t5_ff, fx))
    {
        SKIP("T5 FF fixture pair not found.");
    }
    const std::string full_path =
        t5_fixture_safetensors_path(std::string(T5_DATA_DIR), fx.stem);

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> grad_out_bytes =
        reader.read_tensor("grad_output");
    std::vector<float> grad_out_data(grad_out_bytes.size() / sizeof(float));
    std::memcpy(
        grad_out_data.data(), grad_out_bytes.data(), grad_out_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("grad_input");
    std::vector<float> grad_input_ref(ref_bytes.size() / sizeof(float));
    std::memcpy(grad_input_ref.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> grad_input_result;
    {
        NNGraph g("ff_bwd");
        auto *input =
            g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32, true)
                ->set_name("input");
        T5LayerFF ff(&g, "ff", fx.config);
        ff.load(full_path);
        auto *output = ff.forward(input);

        input->mark_input(true);
        output->mark_output(true);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input, input_data);
        runtime.bind_data(grad_output_tensor, grad_out_data);
        runtime.execute();
        runtime.wait();

        grad_input_result = runtime.get_output<float>(input->grad());
    }

    REQUIRE(grad_input_result.size() == grad_input_ref.size());
    require_relative_frobenius_error(
        grad_input_result, grad_input_ref, fx.backward_tol);
}

#endif
