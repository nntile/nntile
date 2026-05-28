/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/gptneox/gptneox_mlp.cc
 * Tests for GptneoxMlp.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gptneox/gptneox_mlp.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/model/gptneox/gptneox_config.hh"
#include "test_frobenius.hh"
#include "test_gptneox_fixture_helpers.hh"

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
using namespace nntile::graph::model::gptneox;
using namespace nntile::graph::io;

#ifndef GPTNEOX_DATA_DIR

TEST_CASE(
    "GptneoxMlp tests skipped (GPTNEOX_DATA_DIR undefined)", "[model][gptneox]")
{
    SKIP("GPTNEOX_DATA_DIR not defined at compile time.");
}

#else

namespace mlp_fixture_stem
{

constexpr char gptneox_mlp[] = "gptneox_mlp";

} // namespace mlp_fixture_stem

namespace
{

using namespace nntile::graph::test::gptneox_fixture;

struct MlpFixtureSpec
{
    GptneoxConfig config{};
    Index seq = 0;
    Index batch = 0;
    Index hidden = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_mlp_fixture_spec(
    const std::string &data_dir, const char *stem_cstr, MlpFixtureSpec &out)
{
    out = {};
    out.stem = stem_cstr;
    const std::string jpath = data_dir + "/" + out.stem + ".json";
    std::ifstream jf(jpath);
    if (!jf)
    {
        return false;
    }
    nlohmann::json j;
    try
    {
        jf >> j;
        if (j.at("version").get<int>() != 2)
        {
            return false;
        }
        if (j.at("stem").get<std::string>() != out.stem)
        {
            return false;
        }
        const std::string expected_st = out.stem + ".safetensors";
        if (j.at("safetensors").get<std::string>() != expected_st)
        {
            return false;
        }
        const auto &G = j.at("gptneox");
        out.config.hidden_size = json_index(G, "hidden_size");
        out.config.intermediate_size = json_index(G, "intermediate_size");
        out.config.num_attention_heads = json_index(G, "num_attention_heads");
        out.hidden = out.config.hidden_size;
        out.seq = json_index(j, "sequence_length");
        out.batch = json_index(j, "batch");
        out.forward_tol =
            static_cast<float>(j.at("tolerances").at("forward").get<double>());
        out.backward_tol = static_cast<float>(
            j.at("tolerances").at("backward").get<double>());
    }
    catch (...)
    {
        return false;
    }
    prepare_gptneox_config(out.config);
    return true;
}

inline std::string mlp_fixture_safetensors_path(
    const std::string &data_dir, const MlpFixtureSpec &spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

inline bool skip_unless_fixture_ready(const char *stem, MlpFixtureSpec &fx)
{
    const std::string dir = std::string(GPTNEOX_DATA_DIR);
    if (!try_load_mlp_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(mlp_fixture_safetensors_path(dir, fx));
    return st.good();
}

} // namespace

TEST_CASE("GptneoxMlp forward builds output", "[model][gptneox]")
{
    MlpFixtureSpec fx;
    if (!skip_unless_fixture_ready(mlp_fixture_stem::gptneox_mlp, fx))
    {
        SKIP("Missing or invalid gptneox_mlp.json / .safetensors.");
    }
    NNGraph g("gptneox_mlp");
    GptneoxMlp mlp(&g, "mlp", fx.config);
    auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                      ->set_name("input");
    auto *output = mlp.forward(input);

    REQUIRE(output != nullptr);
    REQUIRE(
        output->shape() == std::vector<Index>({fx.hidden, fx.seq, fx.batch}));
    REQUIRE(mlp.parameters_recursive().size() == 4);
}

TEST_CASE("GptneoxMlp load from safetensors roundtrip", "[model][gptneox][io]")
{
    MlpFixtureSpec fx;
    if (!skip_unless_fixture_ready(mlp_fixture_stem::gptneox_mlp, fx))
    {
        SKIP("Missing or invalid gptneox_mlp.json / .safetensors.");
    }
    const std::string data_path =
        mlp_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);

    NNGraph g1("load_graph");
    GptneoxMlp mlp1(&g1, "mlp", fx.config);
    mlp1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_gptneox_mlp_roundtrip.safetensors";
    mlp1.save(save_path);

    SafeTensorsReader reader(data_path);
    SafeTensorsReader reader2(save_path);

    for (const auto &name : reader2.tensor_names())
    {
        REQUIRE(reader.has_tensor(name));
        auto orig = reader.read_tensor(name);
        auto loaded = reader2.read_tensor(name);
        REQUIRE(orig.size() == loaded.size());
        REQUIRE(orig == loaded);
    }

    std::remove(save_path.c_str());
}

TEST_CASE_METHOD(nntile::core::test::ContextFixture,
    "GptneoxMlp matches PyTorch reference",
    "[model][gptneox]")
{
    MlpFixtureSpec fx;
    if (!skip_unless_fixture_ready(mlp_fixture_stem::gptneox_mlp, fx))
    {
        SKIP("GPT-NeoX MLP fixture pair not found.");
    }
    const std::string full_path =
        mlp_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);

    SafeTensorsReader reader(full_path);
    std::vector<std::uint8_t> input_bytes = reader.read_tensor("input");
    std::vector<float> input_data(input_bytes.size() / sizeof(float));
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("mlp_ref");
        auto *input = g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32)
                          ->set_name("input");
        GptneoxMlp mlp(&g, "mlp", fx.config);
        auto *output = mlp.forward(input);
        input->mark_input(true);
        output->mark_output(true);

        mlp.load(full_path);

        TensorGraph &tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

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
    "GptneoxMlp backward matches PyTorch reference",
    "[model][gptneox]")
{
    MlpFixtureSpec fx;
    if (!skip_unless_fixture_ready(mlp_fixture_stem::gptneox_mlp, fx))
    {
        SKIP("GPT-NeoX MLP fixture pair not found.");
    }
    const std::string full_path =
        mlp_fixture_safetensors_path(std::string(GPTNEOX_DATA_DIR), fx);

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
        NNGraph g("mlp_bwd");
        auto *input =
            g.tensor({fx.hidden, fx.seq, fx.batch}, DataType::FP32, true)
                ->set_name("input");
        GptneoxMlp mlp(&g, "mlp", fx.config);
        auto *output = mlp.forward(input);

        input->mark_input(true);
        output->mark_output(true);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        input->grad()->mark_output(true);

        mlp.load(full_path);

        TensorGraph &tg = g.tensor_graph();
        TileGraph tile_graph = TileGraph::from_tensor_graph(tg);

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
