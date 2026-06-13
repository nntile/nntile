/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/gptneo/gptneo_causal.cc
 * Tests for GptneoCausal.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gptneo/gptneo_causal.hh"

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/io/safetensors.hh"
#include "nntile/model/gptneo/gptneo_config.hh"
#include "test_frobenius.hh"
#include "test_gptneo_fixture_helpers.hh"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace nntile;
using namespace nntile;
using namespace nntile::model::gptneo;
using namespace nntile::io;

#ifndef GPTNEO_DATA_DIR

TEST_CASE(
    "GptneoCausal tests skipped (GPTNEO_DATA_DIR undefined)", "[model][gptneo]")
{
    SKIP("GPTNEO_DATA_DIR not defined at compile time.");
}

#else

namespace causal_fixture_stem
{

constexpr char gptneo_causal[] = "gptneo_causal";

} // namespace causal_fixture_stem

namespace
{

using namespace nntile::test::gptneo_fixture;

struct CausalFixtureSpec
{
    GptneoConfig config{};
    Index seq = 0;
    Index batch = 0;
    float forward_tol = 0.f;
    float backward_tol = 0.f;
    std::string stem;
};

inline bool try_load_causal_fixture_spec(const std::string &data_dir,
    const char *stem_cstr,
    CausalFixtureSpec &out)
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
        const auto &G = j.at("gptneo");
        out.config.vocab_size = json_index(G, "vocab_size");
        out.config.hidden_size = json_index(G, "hidden_size");
        out.config.intermediate_size = json_index(G, "intermediate_size");
        out.config.num_hidden_layers = json_index(G, "num_hidden_layers");
        out.config.num_attention_heads = json_index(G, "num_attention_heads");
        out.config.head_dim = json_index(G, "head_dim");
        out.config.max_position_embeddings =
            json_index(G, "max_position_embeddings");
        out.seq = json_index(j, "sequence_length");
        out.batch = json_index(j, "batch");
        out.forward_tol =
            static_cast<float>(j.at("tolerances").at("forward").get<double>());
        out.backward_tol = static_cast<float>(
            j.at("tolerances").at("backward").get<double>());
        out.config.validate();
    }
    catch (...)
    {
        return false;
    }
    prepare_gptneo_config(out.config);
    return true;
}

inline std::string causal_fixture_safetensors_path(
    const std::string &data_dir, const CausalFixtureSpec &spec)
{
    return data_dir + "/" + spec.stem + ".safetensors";
}

inline bool skip_unless_fixture_ready(const char *stem, CausalFixtureSpec &fx)
{
    const std::string dir = std::string(GPTNEO_DATA_DIR);
    if (!try_load_causal_fixture_spec(dir, stem, fx))
    {
        return false;
    }
    std::ifstream st(causal_fixture_safetensors_path(dir, fx));
    return st.good();
}

void causal_backward_compare_ref(const CausalFixtureSpec &fx)
{
    const std::string full_path =
        causal_fixture_safetensors_path(std::string(GPTNEO_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<std::uint8_t> grad_out_bytes =
        reader.read_tensor("grad_output");
    std::vector<float> grad_out_data(grad_out_bytes.size() / sizeof(float));
    std::memcpy(
        grad_out_data.data(), grad_out_bytes.data(), grad_out_bytes.size());

    std::vector<std::uint8_t> ref_wte_bytes =
        reader.read_tensor("grad_wte_vocab");
    std::vector<float> grad_wte_ref(ref_wte_bytes.size() / sizeof(float));
    std::memcpy(
        grad_wte_ref.data(), ref_wte_bytes.data(), ref_wte_bytes.size());

    std::vector<std::uint8_t> ref_wpe_bytes =
        reader.read_tensor("grad_wpe_vocab");
    std::vector<float> grad_wpe_ref(ref_wpe_bytes.size() / sizeof(float));
    std::memcpy(
        grad_wpe_ref.data(), ref_wpe_bytes.data(), ref_wpe_bytes.size());

    std::vector<float> grad_wte_result;
    std::vector<float> grad_wpe_result;
    {
        NNGraph g("causal_bwd");
        auto *input_ids =
            g.tensor({fx.batch, fx.seq}, DataType::INT64, true)
                ->set_name("input_ids");
        NNGraph::TensorNode *position_ids = nullptr;
        std::vector<std::int64_t> pos_data;
        REQUIRE(load_position_ids(
            g, reader, fx.seq, fx.batch, position_ids, pos_data));
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        REQUIRE(load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes));

        GptneoCausal model(&g, "model", fx.config);
        model.load(full_path);
        auto *output = model.forward(input_ids, position_ids, mask);

        input_ids->mark_input(true);
        output->mark_output(true);
        mark_position_ids_input(position_ids);
        mark_mask_input(mask);

        auto [grad_output_tensor, _] =
            g.get_or_create_grad(output, "grad_output");
        grad_output_tensor->mark_input(true);
        output->backward();
        model.model()->wte_vocab_tensor()->grad()->mark_output(true);
        model.model()->wpe_vocab_tensor()->grad()->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input_ids, ids_data);
        runtime.bind_data(grad_output_tensor, grad_out_data);
        bind_position_ids(runtime, position_ids, pos_data);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        grad_wte_result =
            runtime.get_output<float>(model.model()->wte_vocab_tensor()->grad());
        grad_wpe_result =
            runtime.get_output<float>(model.model()->wpe_vocab_tensor()->grad());
    }

    REQUIRE(grad_wte_result.size() == grad_wte_ref.size());
    require_relative_frobenius_error(
        grad_wte_result, grad_wte_ref, fx.backward_tol);
    REQUIRE(grad_wpe_result.size() == grad_wpe_ref.size());
    require_relative_frobenius_error(
        grad_wpe_result, grad_wpe_ref, fx.backward_tol);
}


} // namespace

TEST_CASE("GptneoCausal forward builds output", "[model][gptneo]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::gptneo_causal, fx))
    {
        SKIP("Missing or invalid gptneo_causal.json / .safetensors.");
    }
    NNGraph g("gptneo_causal");
    GptneoCausal model(&g, "model", fx.config);
    auto *input_ids =
        g.tensor({fx.batch, fx.seq}, DataType::INT64)->set_name("input_ids");
    auto *position_ids = g.tensor({fx.batch, fx.seq}, DataType::INT64)
                             ->set_name("position_ids");
    auto *output = model.forward(input_ids, position_ids, nullptr);

    REQUIRE(output != nullptr);
    REQUIRE(output->shape() ==
            std::vector<Index>({fx.config.vocab_size, fx.seq, fx.batch}));
}

TEST_CASE("GptneoCausal load from safetensors roundtrip", "[model][gptneo][io]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::gptneo_causal, fx))
    {
        SKIP("Missing or invalid gptneo_causal.json / .safetensors.");
    }
    const std::string data_path =
        causal_fixture_safetensors_path(std::string(GPTNEO_DATA_DIR), fx);

    NNGraph g1("load_graph");
    GptneoCausal model1(&g1, "model", fx.config);
    model1.load(data_path);

    const std::string save_path =
        "/tmp/nntile_gptneo_causal_roundtrip.safetensors";
    model1.save(save_path);

    SafeTensorsReader reader(data_path);
    SafeTensorsReader reader2(save_path);
    for (const auto &name : reader2.tensor_names())
    {
        REQUIRE(reader.has_tensor(name));
        REQUIRE(reader.read_tensor(name) == reader2.read_tensor(name));
    }
    std::remove(save_path.c_str());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GptneoCausal forward matches PyTorch reference", "[model][gptneo]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::gptneo_causal, fx))
    {
        SKIP("GPT-Neo causal fixture not found.");
    }
    const std::string full_path =
        causal_fixture_safetensors_path(std::string(GPTNEO_DATA_DIR), fx);
    SafeTensorsReader reader(full_path);

    std::vector<std::uint8_t> ids_bytes = reader.read_tensor("input_ids");
    std::vector<std::int64_t> ids_data(ids_bytes.size() / sizeof(std::int64_t));
    std::memcpy(ids_data.data(), ids_bytes.data(), ids_bytes.size());

    std::vector<float> result;
    {
        NNGraph g("causal_ref");
        auto *input_ids =
            g.tensor({fx.batch, fx.seq}, DataType::INT64)->set_name("input_ids");
        NNGraph::TensorNode *position_ids = nullptr;
        std::vector<std::int64_t> pos_data;
        REQUIRE(load_position_ids(
            g, reader, fx.seq, fx.batch, position_ids, pos_data));
        NNGraph::TensorNode *mask = nullptr;
        std::vector<std::uint8_t> mask_bytes;
        REQUIRE(load_attn_mask_bool(g, reader, fx.seq, mask, mask_bytes));

        GptneoCausal model(&g, "model", fx.config);
        model.load(full_path);

        auto *output = model.forward(input_ids, position_ids, mask);
        input_ids->mark_input(true);
        output->mark_output(true);
        mark_position_ids_input(position_ids);
        mark_mask_input(mask);

        TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(input_ids, ids_data);
        bind_position_ids(runtime, position_ids, pos_data);
        bind_mask_input(runtime, mask, mask_bytes);
        runtime.execute();
        runtime.wait();

        result = runtime.get_output<float>(output);
    }

    std::vector<std::uint8_t> ref_bytes = reader.read_tensor("output_ref");
    std::vector<float> ref_data(ref_bytes.size() / sizeof(float));
    std::memcpy(ref_data.data(), ref_bytes.data(), ref_bytes.size());

    REQUIRE(result.size() == ref_data.size());
    require_relative_frobenius_error(result, ref_data, fx.forward_tol);
}


TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GptneoCausal backward matches PyTorch reference",
    "[model][gptneo]")
{
    CausalFixtureSpec fx;
    if (!skip_unless_fixture_ready(causal_fixture_stem::gptneo_causal, fx))
    {
        SKIP("GPT-Neo causal fixture not found.");
    }
    causal_backward_compare_ref(fx);
}

#endif
