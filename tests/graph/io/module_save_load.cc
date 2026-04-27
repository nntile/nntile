/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/io/module_save_load.cc
 * Tests for Module::save() / Module::load().
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/module/embedding.hh"
#include "nntile/graph/module/gated_mlp.hh"
#include "nntile/graph/module/linear.hh"
#include "nntile/graph/module/mlp.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::module;
using namespace nntile::graph::io;

namespace
{

std::string temp_path(const std::string& suffix)
{
    return "/tmp/nntile_test_module_" + suffix + ".safetensors";
}

void remove_temp(const std::string& path)
{
    std::remove(path.c_str());
}

std::vector<std::uint8_t> make_fp32_bytes(const std::vector<float>& data)
{
    std::vector<std::uint8_t> bytes(data.size() * sizeof(float));
    std::memcpy(bytes.data(), data.data(), bytes.size());
    return bytes;
}

std::vector<float> bytes_to_fp32(const std::vector<std::uint8_t>& bytes)
{
    std::vector<float> result(bytes.size() / sizeof(float));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

} // anonymous namespace

// -------------------------------------------------------------------------
// Linear save/load round-trip
// -------------------------------------------------------------------------

TEST_CASE("Linear save and load round-trip", "[io][module]")
{
    const std::string path = temp_path("linear_save_load");
    const Index in_dim = 3;
    const Index out_dim = 4;

    // Create and populate Linear
    NNGraph g1("save_graph");
    Linear linear1(&g1, "linear", in_dim, out_dim, true);

    std::vector<float> w_data(in_dim * out_dim);
    for(Index i = 0; i < in_dim * out_dim; ++i)
    {
        w_data[i] = 0.1f * static_cast<float>(i + 1);
    }
    linear1.bind_weight(w_data);

    std::vector<float> b_data(out_dim);
    for(Index i = 0; i < out_dim; ++i)
    {
        b_data[i] = static_cast<float>(i + 1);
    }
    linear1.bind_bias(b_data);

    // Save
    linear1.save(path);

    // Verify file contents via reader
    {
        SafeTensorsReader reader(path);
        REQUIRE(reader.size() == 2);
        REQUIRE(reader.has_tensor("linear.weight"));
        REQUIRE(reader.has_tensor("linear.bias"));

        const auto& w_info = reader.tensor_info("linear.weight");
        REQUIRE(w_info.shape == std::vector<std::int64_t>({in_dim, out_dim}));
        REQUIRE(w_info.dtype == DataType::FP32);

        const auto& b_info = reader.tensor_info("linear.bias");
        REQUIRE(b_info.shape == std::vector<std::int64_t>({out_dim}));
    }

    // Load into a new Linear
    NNGraph g2("load_graph");
    Linear linear2(&g2, "linear", in_dim, out_dim, true);
    linear2.load(path);

    // Verify loaded data matches
    const auto* w_hint = linear2.weight_tensor()->data()->get_bind_hint();
    REQUIRE(w_hint != nullptr);
    auto w_loaded = bytes_to_fp32(*w_hint);
    REQUIRE(w_loaded.size() == w_data.size());
    for(std::size_t i = 0; i < w_data.size(); ++i)
    {
        REQUIRE(w_loaded[i] == w_data[i]);
    }

    const auto* b_hint = linear2.bias_tensor()->data()->get_bind_hint();
    REQUIRE(b_hint != nullptr);
    auto b_loaded = bytes_to_fp32(*b_hint);
    REQUIRE(b_loaded.size() == b_data.size());
    for(std::size_t i = 0; i < b_data.size(); ++i)
    {
        REQUIRE(b_loaded[i] == b_data[i]);
    }

    remove_temp(path);
}

TEST_CASE("Linear save without bias", "[io][module]")
{
    const std::string path = temp_path("linear_no_bias");

    NNGraph g("graph");
    Linear linear(&g, "linear", 3, 4, false);

    std::vector<float> w_data(12, 1.0f);
    linear.bind_weight(w_data);
    linear.save(path);

    SafeTensorsReader reader(path);
    REQUIRE(reader.size() == 1);
    REQUIRE(reader.has_tensor("linear.weight"));
    REQUIRE_FALSE(reader.has_tensor("linear.bias"));

    remove_temp(path);
}

// -------------------------------------------------------------------------
// Module load strict mode
// -------------------------------------------------------------------------

TEST_CASE("Module load strict mode throws on missing tensor", "[io][module]")
{
    const std::string path = temp_path("strict_mode");

    // Save only weight (no bias)
    {
        SafeTensorsWriter writer;
        std::vector<std::uint8_t> data(3 * 4 * sizeof(float), 0);
        writer.add_tensor("linear.weight", DataType::FP32, {3, 4}, data);
        writer.write(path);
    }

    // Load into a module that has bias -> strict should fail
    NNGraph g("graph");
    Linear linear(&g, "linear", 3, 4, true);

    REQUIRE_THROWS_AS(
        linear.load(path, true),
        std::runtime_error);

    // Non-strict should succeed
    linear.load(path, false);

    remove_temp(path);
}

// -------------------------------------------------------------------------
// Module load dtype mismatch
// -------------------------------------------------------------------------

TEST_CASE("Module load rejects dtype mismatch", "[io][module]")
{
    const std::string path = temp_path("dtype_mismatch");

    // Save as FP64
    {
        SafeTensorsWriter writer;
        std::vector<std::uint8_t> data(3 * 4 * sizeof(double), 0);
        writer.add_tensor("linear.weight", DataType::FP64, {3, 4}, data);
        writer.write(path);
    }

    // Load into FP32 module -> should fail
    NNGraph g("graph");
    Linear linear(&g, "linear", 3, 4, false, DataType::FP32);

    REQUIRE_THROWS_AS(
        linear.load(path),
        std::runtime_error);

    remove_temp(path);
}

// -------------------------------------------------------------------------
// Module load shape mismatch
// -------------------------------------------------------------------------

TEST_CASE("Module load rejects shape mismatch", "[io][module]")
{
    const std::string path = temp_path("shape_mismatch");

    // Save as (5, 4) weight
    {
        SafeTensorsWriter writer;
        std::vector<std::uint8_t> data(5 * 4 * sizeof(float), 0);
        writer.add_tensor("linear.weight", DataType::FP32, {5, 4}, data);
        writer.write(path);
    }

    // Load into (3, 4) module -> should fail
    NNGraph g("graph");
    Linear linear(&g, "linear", 3, 4, false);

    REQUIRE_THROWS_AS(
        linear.load(path),
        std::runtime_error);

    remove_temp(path);
}

// =========================================================================
// GatedMlp tests (nested module with 3 Linear children)
// =========================================================================

TEST_CASE("GatedMlp save and load round-trip", "[io][module][gated_mlp]")
{
    const std::string path = temp_path("gated_mlp_save_load");
    const Index in_dim = 4;
    const Index inter_dim = 8;
    const Index out_dim = 4;

    // Build and populate GatedMlp
    NNGraph g1("save_graph");
    GatedMlp gmlp1(&g1, "gmlp", in_dim, inter_dim, out_dim);

    // Bind data to all 3 Linear weights
    std::vector<float> gate_w(in_dim * inter_dim);
    std::vector<float> up_w(in_dim * inter_dim);
    std::vector<float> down_w(inter_dim * out_dim);
    for(std::size_t i = 0; i < gate_w.size(); ++i)
        gate_w[i] = 0.01f * static_cast<float>(i);
    for(std::size_t i = 0; i < up_w.size(); ++i)
        up_w[i] = 0.02f * static_cast<float>(i);
    for(std::size_t i = 0; i < down_w.size(); ++i)
        down_w[i] = 0.03f * static_cast<float>(i);

    gmlp1.gate_proj().bind_weight(gate_w);
    gmlp1.up_proj().bind_weight(up_w);
    gmlp1.down_proj().bind_weight(down_w);

    gmlp1.save(path);

    // Verify the file has the correct nested names
    {
        SafeTensorsReader reader(path);
        REQUIRE(reader.size() == 3);
        REQUIRE(reader.has_tensor("gmlp.gate_proj.weight"));
        REQUIRE(reader.has_tensor("gmlp.up_proj.weight"));
        REQUIRE(reader.has_tensor("gmlp.down_proj.weight"));

        REQUIRE(reader.tensor_info("gmlp.gate_proj.weight").shape ==
                std::vector<std::int64_t>({in_dim, inter_dim}));
        REQUIRE(reader.tensor_info("gmlp.up_proj.weight").shape ==
                std::vector<std::int64_t>({in_dim, inter_dim}));
        REQUIRE(reader.tensor_info("gmlp.down_proj.weight").shape ==
                std::vector<std::int64_t>({inter_dim, out_dim}));
    }

    // Load into a new GatedMlp
    NNGraph g2("load_graph");
    GatedMlp gmlp2(&g2, "gmlp", in_dim, inter_dim, out_dim);
    gmlp2.load(path);

    // Verify gate_proj weight
    const auto* gate_hint = gmlp2.gate_proj().weight_tensor()->data()->get_bind_hint();
    REQUIRE(gate_hint != nullptr);
    auto gate_loaded = bytes_to_fp32(*gate_hint);
    REQUIRE(gate_loaded.size() == gate_w.size());
    for(std::size_t i = 0; i < gate_w.size(); ++i)
        REQUIRE(gate_loaded[i] == gate_w[i]);

    // Verify up_proj weight
    const auto* up_hint = gmlp2.up_proj().weight_tensor()->data()->get_bind_hint();
    REQUIRE(up_hint != nullptr);
    auto up_loaded = bytes_to_fp32(*up_hint);
    for(std::size_t i = 0; i < up_w.size(); ++i)
        REQUIRE(up_loaded[i] == up_w[i]);

    // Verify down_proj weight
    const auto* down_hint = gmlp2.down_proj().weight_tensor()->data()->get_bind_hint();
    REQUIRE(down_hint != nullptr);
    auto down_loaded = bytes_to_fp32(*down_hint);
    for(std::size_t i = 0; i < down_w.size(); ++i)
        REQUIRE(down_loaded[i] == down_w[i]);

    remove_temp(path);
}

// =========================================================================
// Mlp tests (nested module with 2 Linear children)
// =========================================================================

TEST_CASE("Mlp save and load round-trip", "[io][module][mlp]")
{
    const std::string path = temp_path("mlp_save_load");
    const Index in_dim = 4;
    const Index inter_dim = 8;
    const Index out_dim = 4;

    NNGraph g1("save_graph");
    Mlp mlp1(&g1, "mlp", in_dim, inter_dim, out_dim);

    std::vector<float> fc1_w(in_dim * inter_dim);
    std::vector<float> fc2_w(inter_dim * out_dim);
    for(std::size_t i = 0; i < fc1_w.size(); ++i)
        fc1_w[i] = 0.05f * static_cast<float>(i);
    for(std::size_t i = 0; i < fc2_w.size(); ++i)
        fc2_w[i] = 0.06f * static_cast<float>(i);

    mlp1.fc1().bind_weight(fc1_w);
    mlp1.fc2().bind_weight(fc2_w);
    mlp1.save(path);

    // Verify nested names
    {
        SafeTensorsReader reader(path);
        REQUIRE(reader.size() == 2);
        REQUIRE(reader.has_tensor("mlp.fc1.weight"));
        REQUIRE(reader.has_tensor("mlp.fc2.weight"));
    }

    // Load
    NNGraph g2("load_graph");
    Mlp mlp2(&g2, "mlp", in_dim, inter_dim, out_dim);
    mlp2.load(path);

    const auto* fc1_hint = mlp2.fc1().weight_tensor()->data()->get_bind_hint();
    REQUIRE(fc1_hint != nullptr);
    auto fc1_loaded = bytes_to_fp32(*fc1_hint);
    for(std::size_t i = 0; i < fc1_w.size(); ++i)
        REQUIRE(fc1_loaded[i] == fc1_w[i]);

    const auto* fc2_hint = mlp2.fc2().weight_tensor()->data()->get_bind_hint();
    REQUIRE(fc2_hint != nullptr);
    auto fc2_loaded = bytes_to_fp32(*fc2_hint);
    for(std::size_t i = 0; i < fc2_w.size(); ++i)
        REQUIRE(fc2_loaded[i] == fc2_w[i]);

    remove_temp(path);
}

// =========================================================================
// Embedding tests
// =========================================================================

TEST_CASE("Embedding save and load round-trip", "[io][module][embedding]")
{
    const std::string path = temp_path("embedding_save_load");
    const Index num_emb = 100;
    const Index emb_dim = 16;

    NNGraph g1("save_graph");
    Embedding emb1(&g1, "embed", num_emb, emb_dim);

    // NNTile stores vocab as (embed_dim, num_embeddings) column-major
    std::vector<float> vocab(emb_dim * num_emb);
    for(std::size_t i = 0; i < vocab.size(); ++i)
        vocab[i] = 0.001f * static_cast<float>(i);
    emb1.bind_weight(vocab);
    emb1.save(path);

    {
        SafeTensorsReader reader(path);
        REQUIRE(reader.size() == 1);
        REQUIRE(reader.has_tensor("embed.vocab"));
        REQUIRE(reader.tensor_info("embed.vocab").shape ==
                std::vector<std::int64_t>({emb_dim, num_emb}));
    }

    NNGraph g2("load_graph");
    Embedding emb2(&g2, "embed", num_emb, emb_dim);
    emb2.load(path);

    const auto* hint = emb2.vocab_tensor()->data()->get_bind_hint();
    REQUIRE(hint != nullptr);
    auto loaded = bytes_to_fp32(*hint);
    REQUIRE(loaded.size() == vocab.size());
    for(std::size_t i = 0; i < vocab.size(); ++i)
        REQUIRE(loaded[i] == vocab[i]);

    remove_temp(path);
}
