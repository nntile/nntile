/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/io/module_save_load.cc
 * Tests for Module::save() / Module::load() and Linear HF import/export.
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
#include "nntile/io/safetensors.hh"
#include "nntile/module/embedding.hh"
#include "nntile/module/gated_mlp.hh"
#include "nntile/module/linear.hh"
#include "nntile/module/mlp.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::module;
using namespace nntile::io;

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
// Linear HF import/export
// -------------------------------------------------------------------------

TEST_CASE("Linear import_hf transposes weight", "[io][module][hf]")
{
    const std::string path = temp_path("linear_hf_import");
    const Index in_dim = 3;
    const Index out_dim = 4;

    // Create an HF-format safetensors file where weight is (out, in) row-major.
    // In HF row-major, weight[j][i] is stored at offset (j * in + i).
    std::vector<float> hf_weight(out_dim * in_dim);
    for(std::int64_t j = 0; j < out_dim; ++j)
    {
        for(std::int64_t i = 0; i < in_dim; ++i)
        {
            hf_weight[static_cast<std::size_t>(j * in_dim + i)] =
                static_cast<float>(j * 10 + i);
        }
    }
    std::vector<float> hf_bias = {100.0f, 200.0f, 300.0f, 400.0f};

    {
        SafeTensorsWriter writer;
        writer.add_tensor(
            "layer.weight", DataType::FP32,
            {out_dim, in_dim},
            make_fp32_bytes(hf_weight));
        writer.add_tensor(
            "layer.bias", DataType::FP32,
            {out_dim},
            make_fp32_bytes(hf_bias));
        writer.write(path);
    }

    // Import into NNTile Linear
    NNGraph g("graph");
    Linear linear(&g, "linear", in_dim, out_dim, true);

    SafeTensorsReader reader(path);
    linear.import_hf(reader, "layer");

    // Verify the weight was transposed correctly.
    // NNTile stores (in_dim, out_dim) column-major: nntile[i + j*in_dim]
    // should equal hf[j][i] = j*10 + i.
    const auto* w_hint = linear.weight_tensor()->data()->get_bind_hint();
    REQUIRE(w_hint != nullptr);
    auto w_loaded = bytes_to_fp32(*w_hint);
    REQUIRE(w_loaded.size() == static_cast<std::size_t>(in_dim * out_dim));

    for(Index j = 0; j < out_dim; ++j)
    {
        for(Index i = 0; i < in_dim; ++i)
        {
            float expected = static_cast<float>(j * 10 + i);
            float actual = w_loaded[static_cast<std::size_t>(i + j * in_dim)];
            REQUIRE(actual == expected);
        }
    }

    // Bias: should be unchanged
    const auto* b_hint = linear.bias_tensor()->data()->get_bind_hint();
    REQUIRE(b_hint != nullptr);
    auto b_loaded = bytes_to_fp32(*b_hint);
    REQUIRE(b_loaded == hf_bias);

    remove_temp(path);
}

TEST_CASE("Linear export_hf transposes weight back", "[io][module][hf]")
{
    const std::string path = temp_path("linear_hf_export");
    const Index in_dim = 3;
    const Index out_dim = 4;

    // Create a Linear with NNTile-format weight: (in_dim, out_dim) col-major
    // nntile[i + j*in_dim] = j*10 + i
    NNGraph g("graph");
    Linear linear(&g, "linear", in_dim, out_dim, true);

    std::vector<float> w_nntile(in_dim * out_dim);
    for(Index j = 0; j < out_dim; ++j)
    {
        for(Index i = 0; i < in_dim; ++i)
        {
            w_nntile[static_cast<std::size_t>(i + j * in_dim)] =
                static_cast<float>(j * 10 + i);
        }
    }
    linear.bind_weight(w_nntile);

    std::vector<float> bias = {100.0f, 200.0f, 300.0f, 400.0f};
    linear.bind_bias(bias);

    // Export to HF format
    SafeTensorsWriter writer;
    linear.export_hf(writer, "layer");
    writer.write(path);

    // Read back and verify HF layout: (out_dim, in_dim) row-major
    SafeTensorsReader reader(path);
    REQUIRE(reader.has_tensor("layer.weight"));
    REQUIRE(reader.has_tensor("layer.bias"));

    const auto& w_info = reader.tensor_info("layer.weight");
    REQUIRE(w_info.shape == std::vector<std::int64_t>({out_dim, in_dim}));

    auto w_bytes = reader.read_tensor("layer.weight");
    auto w_hf = bytes_to_fp32(w_bytes);

    for(Index j = 0; j < out_dim; ++j)
    {
        for(Index i = 0; i < in_dim; ++i)
        {
            float expected = static_cast<float>(j * 10 + i);
            float actual = w_hf[static_cast<std::size_t>(j * in_dim + i)];
            REQUIRE(actual == expected);
        }
    }

    auto b_bytes = reader.read_tensor("layer.bias");
    auto b_hf = bytes_to_fp32(b_bytes);
    REQUIRE(b_hf == bias);

    remove_temp(path);
}

TEST_CASE("Linear HF import then export round-trip", "[io][module][hf]")
{
    const std::string path_in = temp_path("hf_roundtrip_in");
    const std::string path_out = temp_path("hf_roundtrip_out");
    const Index in_dim = 5;
    const Index out_dim = 3;

    // Create original HF data
    std::vector<float> hf_weight(out_dim * in_dim);
    for(std::size_t i = 0; i < hf_weight.size(); ++i)
    {
        hf_weight[i] = static_cast<float>(i) * 0.01f;
    }

    {
        SafeTensorsWriter writer;
        writer.add_tensor(
            "fc.weight", DataType::FP32,
            {out_dim, in_dim},
            make_fp32_bytes(hf_weight));
        writer.write(path_in);
    }

    // Import
    NNGraph g("graph");
    Linear linear(&g, "linear", in_dim, out_dim, false);
    {
        SafeTensorsReader reader(path_in);
        linear.import_hf(reader, "fc");
    }

    // Export
    {
        SafeTensorsWriter writer;
        linear.export_hf(writer, "fc");
        writer.write(path_out);
    }

    // Verify exported matches original
    {
        SafeTensorsReader reader(path_out);
        auto loaded = reader.read_tensor("fc.weight");
        auto loaded_floats = bytes_to_fp32(loaded);
        REQUIRE(loaded_floats.size() == hf_weight.size());
        for(std::size_t i = 0; i < hf_weight.size(); ++i)
        {
            REQUIRE(loaded_floats[i] == hf_weight[i]);
        }
    }

    remove_temp(path_in);
    remove_temp(path_out);
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

TEST_CASE("GatedMlp HF import via default delegation", "[io][module][gated_mlp][hf]")
{
    const std::string path = temp_path("gated_mlp_hf_import");
    const Index in_dim = 4;
    const Index inter_dim = 8;
    const Index out_dim = 4;

    // Create HF-format safetensors with Llama-style naming
    // HF stores Linear weight as (out_features, in_features) row-major
    std::vector<float> hf_gate_w(inter_dim * in_dim);
    std::vector<float> hf_up_w(inter_dim * in_dim);
    std::vector<float> hf_down_w(out_dim * inter_dim);
    for(std::size_t i = 0; i < hf_gate_w.size(); ++i)
        hf_gate_w[i] = 0.1f * static_cast<float>(i);
    for(std::size_t i = 0; i < hf_up_w.size(); ++i)
        hf_up_w[i] = 0.2f * static_cast<float>(i);
    for(std::size_t i = 0; i < hf_down_w.size(); ++i)
        hf_down_w[i] = 0.3f * static_cast<float>(i);

    {
        SafeTensorsWriter writer;
        writer.add_tensor(
            "layers.0.mlp.gate_proj.weight", DataType::FP32,
            {inter_dim, in_dim}, make_fp32_bytes(hf_gate_w));
        writer.add_tensor(
            "layers.0.mlp.up_proj.weight", DataType::FP32,
            {inter_dim, in_dim}, make_fp32_bytes(hf_up_w));
        writer.add_tensor(
            "layers.0.mlp.down_proj.weight", DataType::FP32,
            {out_dim, inter_dim}, make_fp32_bytes(hf_down_w));
        writer.write(path);
    }

    // Import via default delegation
    NNGraph g("graph");
    GatedMlp gmlp(&g, "gmlp", in_dim, inter_dim, out_dim);

    SafeTensorsReader reader(path);
    gmlp.import_hf(reader, "layers.0.mlp");

    // Verify each child got its data.
    // HF (out, in) row-major == NNTile (in, out) column-major, same bytes.
    const auto* gate_hint = gmlp.gate_proj().weight_tensor()->data()->get_bind_hint();
    REQUIRE(gate_hint != nullptr);
    auto gate_loaded = bytes_to_fp32(*gate_hint);
    REQUIRE(gate_loaded.size() == hf_gate_w.size());
    for(std::size_t i = 0; i < hf_gate_w.size(); ++i)
        REQUIRE(gate_loaded[i] == hf_gate_w[i]);

    const auto* up_hint = gmlp.up_proj().weight_tensor()->data()->get_bind_hint();
    REQUIRE(up_hint != nullptr);
    auto up_loaded = bytes_to_fp32(*up_hint);
    for(std::size_t i = 0; i < hf_up_w.size(); ++i)
        REQUIRE(up_loaded[i] == hf_up_w[i]);

    const auto* down_hint = gmlp.down_proj().weight_tensor()->data()->get_bind_hint();
    REQUIRE(down_hint != nullptr);
    auto down_loaded = bytes_to_fp32(*down_hint);
    for(std::size_t i = 0; i < hf_down_w.size(); ++i)
        REQUIRE(down_loaded[i] == hf_down_w[i]);

    remove_temp(path);
}

TEST_CASE("GatedMlp HF export via default delegation", "[io][module][gated_mlp][hf]")
{
    const std::string path = temp_path("gated_mlp_hf_export");
    const Index in_dim = 4;
    const Index inter_dim = 8;
    const Index out_dim = 4;

    NNGraph g("graph");
    GatedMlp gmlp(&g, "gmlp", in_dim, inter_dim, out_dim);

    std::vector<float> gate_w(in_dim * inter_dim, 1.0f);
    std::vector<float> up_w(in_dim * inter_dim, 2.0f);
    std::vector<float> down_w(inter_dim * out_dim, 3.0f);
    gmlp.gate_proj().bind_weight(gate_w);
    gmlp.up_proj().bind_weight(up_w);
    gmlp.down_proj().bind_weight(down_w);

    SafeTensorsWriter writer;
    gmlp.export_hf(writer, "model.layers.0.mlp");
    writer.write(path);

    SafeTensorsReader reader(path);
    REQUIRE(reader.size() == 3);
    REQUIRE(reader.has_tensor("model.layers.0.mlp.gate_proj.weight"));
    REQUIRE(reader.has_tensor("model.layers.0.mlp.up_proj.weight"));
    REQUIRE(reader.has_tensor("model.layers.0.mlp.down_proj.weight"));

    // HF shape is (out_features, in_features) = (inter_dim, in_dim)
    REQUIRE(reader.tensor_info("model.layers.0.mlp.gate_proj.weight").shape ==
            std::vector<std::int64_t>({inter_dim, in_dim}));
    REQUIRE(reader.tensor_info("model.layers.0.mlp.down_proj.weight").shape ==
            std::vector<std::int64_t>({out_dim, inter_dim}));

    remove_temp(path);
}

TEST_CASE("GatedMlp HF import then export round-trip", "[io][module][gated_mlp][hf]")
{
    const std::string path_in = temp_path("gated_mlp_hf_rt_in");
    const std::string path_out = temp_path("gated_mlp_hf_rt_out");
    const Index in_dim = 4;
    const Index inter_dim = 6;

    // Create HF data
    std::vector<float> hf_gate(inter_dim * in_dim);
    std::vector<float> hf_up(inter_dim * in_dim);
    std::vector<float> hf_down(in_dim * inter_dim);
    for(std::size_t i = 0; i < hf_gate.size(); ++i)
        hf_gate[i] = static_cast<float>(i) * 0.001f;
    for(std::size_t i = 0; i < hf_up.size(); ++i)
        hf_up[i] = static_cast<float>(i) * 0.002f;
    for(std::size_t i = 0; i < hf_down.size(); ++i)
        hf_down[i] = static_cast<float>(i) * 0.003f;

    {
        SafeTensorsWriter writer;
        writer.add_tensor("mlp.gate_proj.weight", DataType::FP32,
                          {inter_dim, in_dim}, make_fp32_bytes(hf_gate));
        writer.add_tensor("mlp.up_proj.weight", DataType::FP32,
                          {inter_dim, in_dim}, make_fp32_bytes(hf_up));
        writer.add_tensor("mlp.down_proj.weight", DataType::FP32,
                          {in_dim, inter_dim}, make_fp32_bytes(hf_down));
        writer.write(path_in);
    }

    // Import
    NNGraph g("graph");
    GatedMlp gmlp(&g, "gmlp", in_dim, inter_dim, in_dim);
    {
        SafeTensorsReader reader(path_in);
        gmlp.import_hf(reader, "mlp");
    }

    // Export
    {
        SafeTensorsWriter writer;
        gmlp.export_hf(writer, "mlp");
        writer.write(path_out);
    }

    // Verify round-trip: exported bytes match original
    {
        SafeTensorsReader reader(path_out);
        auto gate_bytes = reader.read_tensor("mlp.gate_proj.weight");
        auto gate_floats = bytes_to_fp32(gate_bytes);
        REQUIRE(gate_floats.size() == hf_gate.size());
        for(std::size_t i = 0; i < hf_gate.size(); ++i)
            REQUIRE(gate_floats[i] == hf_gate[i]);

        auto up_bytes = reader.read_tensor("mlp.up_proj.weight");
        auto up_floats = bytes_to_fp32(up_bytes);
        for(std::size_t i = 0; i < hf_up.size(); ++i)
            REQUIRE(up_floats[i] == hf_up[i]);

        auto down_bytes = reader.read_tensor("mlp.down_proj.weight");
        auto down_floats = bytes_to_fp32(down_bytes);
        for(std::size_t i = 0; i < hf_down.size(); ++i)
            REQUIRE(down_floats[i] == hf_down[i]);
    }

    remove_temp(path_in);
    remove_temp(path_out);
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

TEST_CASE("Mlp HF import and export round-trip", "[io][module][mlp][hf]")
{
    const std::string path_in = temp_path("mlp_hf_rt_in");
    const std::string path_out = temp_path("mlp_hf_rt_out");
    const Index in_dim = 4;
    const Index inter_dim = 8;

    std::vector<float> hf_fc1(inter_dim * in_dim);
    std::vector<float> hf_fc2(in_dim * inter_dim);
    for(std::size_t i = 0; i < hf_fc1.size(); ++i)
        hf_fc1[i] = static_cast<float>(i) * 0.01f;
    for(std::size_t i = 0; i < hf_fc2.size(); ++i)
        hf_fc2[i] = static_cast<float>(i) * 0.02f;

    {
        SafeTensorsWriter writer;
        writer.add_tensor("block.mlp.fc1.weight", DataType::FP32,
                          {inter_dim, in_dim}, make_fp32_bytes(hf_fc1));
        writer.add_tensor("block.mlp.fc2.weight", DataType::FP32,
                          {in_dim, inter_dim}, make_fp32_bytes(hf_fc2));
        writer.write(path_in);
    }

    NNGraph g("graph");
    Mlp mlp(&g, "mlp", in_dim, inter_dim, in_dim);
    {
        SafeTensorsReader reader(path_in);
        mlp.import_hf(reader, "block.mlp");
    }

    {
        SafeTensorsWriter writer;
        mlp.export_hf(writer, "block.mlp");
        writer.write(path_out);
    }

    {
        SafeTensorsReader reader(path_out);
        auto fc1_loaded = bytes_to_fp32(reader.read_tensor("block.mlp.fc1.weight"));
        REQUIRE(fc1_loaded.size() == hf_fc1.size());
        for(std::size_t i = 0; i < hf_fc1.size(); ++i)
            REQUIRE(fc1_loaded[i] == hf_fc1[i]);

        auto fc2_loaded = bytes_to_fp32(reader.read_tensor("block.mlp.fc2.weight"));
        for(std::size_t i = 0; i < hf_fc2.size(); ++i)
            REQUIRE(fc2_loaded[i] == hf_fc2[i]);
    }

    remove_temp(path_in);
    remove_temp(path_out);
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

TEST_CASE("Embedding HF import transposes dimensions", "[io][module][embedding][hf]")
{
    const std::string path = temp_path("embedding_hf_import");
    const Index num_emb = 50;
    const Index emb_dim = 8;

    // HF embedding weight: (num_embeddings, embed_dim) row-major
    std::vector<float> hf_weight(num_emb * emb_dim);
    for(std::size_t i = 0; i < hf_weight.size(); ++i)
        hf_weight[i] = static_cast<float>(i) * 0.01f;

    {
        SafeTensorsWriter writer;
        writer.add_tensor("model.embed_tokens.weight", DataType::FP32,
                          {num_emb, emb_dim}, make_fp32_bytes(hf_weight));
        writer.write(path);
    }

    NNGraph g("graph");
    Embedding emb(&g, "embed", num_emb, emb_dim);

    SafeTensorsReader reader(path);
    emb.import_hf(reader, "model.embed_tokens");

    // HF row-major (num_emb, emb_dim) == NNTile col-major (emb_dim, num_emb)
    const auto* hint = emb.vocab_tensor()->data()->get_bind_hint();
    REQUIRE(hint != nullptr);
    auto loaded = bytes_to_fp32(*hint);
    REQUIRE(loaded.size() == hf_weight.size());
    for(std::size_t i = 0; i < hf_weight.size(); ++i)
        REQUIRE(loaded[i] == hf_weight[i]);

    remove_temp(path);
}

TEST_CASE("Embedding HF export writes correct shape", "[io][module][embedding][hf]")
{
    const std::string path = temp_path("embedding_hf_export");
    const Index num_emb = 50;
    const Index emb_dim = 8;

    NNGraph g("graph");
    Embedding emb(&g, "embed", num_emb, emb_dim);

    std::vector<float> vocab(emb_dim * num_emb, 0.5f);
    emb.bind_weight(vocab);

    SafeTensorsWriter writer;
    emb.export_hf(writer, "model.embed_tokens");
    writer.write(path);

    SafeTensorsReader reader(path);
    REQUIRE(reader.has_tensor("model.embed_tokens.weight"));
    // HF shape: (num_embeddings, embed_dim)
    REQUIRE(reader.tensor_info("model.embed_tokens.weight").shape ==
            std::vector<std::int64_t>({num_emb, emb_dim}));

    remove_temp(path);
}

TEST_CASE("Embedding HF import then export round-trip", "[io][module][embedding][hf]")
{
    const std::string path_in = temp_path("embedding_hf_rt_in");
    const std::string path_out = temp_path("embedding_hf_rt_out");
    const Index num_emb = 30;
    const Index emb_dim = 12;

    std::vector<float> hf_weight(num_emb * emb_dim);
    for(std::size_t i = 0; i < hf_weight.size(); ++i)
        hf_weight[i] = static_cast<float>(i) * 0.005f;

    {
        SafeTensorsWriter writer;
        writer.add_tensor("emb.weight", DataType::FP32,
                          {num_emb, emb_dim}, make_fp32_bytes(hf_weight));
        writer.write(path_in);
    }

    NNGraph g("graph");
    Embedding emb(&g, "embed", num_emb, emb_dim);
    {
        SafeTensorsReader reader(path_in);
        emb.import_hf(reader, "emb");
    }

    {
        SafeTensorsWriter writer;
        emb.export_hf(writer, "emb");
        writer.write(path_out);
    }

    {
        SafeTensorsReader reader(path_out);
        auto loaded = bytes_to_fp32(reader.read_tensor("emb.weight"));
        REQUIRE(loaded.size() == hf_weight.size());
        for(std::size_t i = 0; i < hf_weight.size(); ++i)
            REQUIRE(loaded[i] == hf_weight[i]);
    }

    remove_temp(path_in);
    remove_temp(path_out);
}
