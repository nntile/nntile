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
