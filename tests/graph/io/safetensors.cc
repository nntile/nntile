/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/io/safetensors.cc
 * Tests for SafeTensors reader/writer and dtype mapping.
 *
 * @version 1.1.0
 * */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph/io/safetensors.hh"
#include "nntile/graph/dtype.hh"

using namespace nntile;
using namespace nntile::graph::io;
using namespace nntile::graph;

namespace
{

std::string temp_path(const std::string& suffix)
{
    return "/tmp/nntile_test_safetensors_" + suffix + ".safetensors";
}

void remove_temp(const std::string& path)
{
    std::remove(path.c_str());
}

} // anonymous namespace

// -------------------------------------------------------------------------
// Dtype mapping tests
// -------------------------------------------------------------------------

TEST_CASE("dtype_to_safetensors maps correctly", "[io][dtype]")
{
    REQUIRE(dtype_to_safetensors(DataType::FP32) == "F32");
    REQUIRE(dtype_to_safetensors(DataType::FP32_FAST_TF32) == "F32");
    REQUIRE(dtype_to_safetensors(DataType::FP32_FAST_FP16) == "F32");
    REQUIRE(dtype_to_safetensors(DataType::FP32_FAST_BF16) == "F32");
    REQUIRE(dtype_to_safetensors(DataType::FP64) == "F64");
    REQUIRE(dtype_to_safetensors(DataType::FP16) == "F16");
    REQUIRE(dtype_to_safetensors(DataType::BF16) == "BF16");
    REQUIRE(dtype_to_safetensors(DataType::INT64) == "I64");
    REQUIRE(dtype_to_safetensors(DataType::BOOL) == "BOOL");
}

TEST_CASE("safetensors_to_dtype maps correctly", "[io][dtype]")
{
    REQUIRE(safetensors_to_dtype("F32") == DataType::FP32);
    REQUIRE(safetensors_to_dtype("F64") == DataType::FP64);
    REQUIRE(safetensors_to_dtype("F16") == DataType::FP16);
    REQUIRE(safetensors_to_dtype("BF16") == DataType::BF16);
    REQUIRE(safetensors_to_dtype("I64") == DataType::INT64);
    REQUIRE(safetensors_to_dtype("BOOL") == DataType::BOOL);

    REQUIRE_THROWS_AS(safetensors_to_dtype("UNKNOWN"), std::invalid_argument);
}

TEST_CASE("is_safetensors_dtype_compatible checks correctly", "[io][dtype]")
{
    REQUIRE(is_safetensors_dtype_compatible("F32", DataType::FP32));
    REQUIRE(is_safetensors_dtype_compatible("F32", DataType::FP32_FAST_TF32));
    REQUIRE(is_safetensors_dtype_compatible("F32", DataType::FP32_FAST_FP16));
    REQUIRE(is_safetensors_dtype_compatible("F32", DataType::FP32_FAST_BF16));
    REQUIRE_FALSE(is_safetensors_dtype_compatible("F64", DataType::FP32));
    REQUIRE_FALSE(is_safetensors_dtype_compatible("F32", DataType::FP64));
}

// -------------------------------------------------------------------------
// Writer/Reader round-trip tests
// -------------------------------------------------------------------------

TEST_CASE("SafeTensors single FP32 tensor round-trip", "[io][safetensors]")
{
    const std::string path = temp_path("single_fp32");

    // Create test data: 3x4 FP32 tensor
    std::vector<float> original(12);
    for(int i = 0; i < 12; ++i)
    {
        original[i] = 0.1f * static_cast<float>(i + 1);
    }
    std::vector<std::uint8_t> bytes(original.size() * sizeof(float));
    std::memcpy(bytes.data(), original.data(), bytes.size());

    // Write
    {
        SafeTensorsWriter writer;
        writer.add_tensor("weight", DataType::FP32, {3, 4}, bytes);
        REQUIRE(writer.size() == 1);
        writer.write(path);
    }

    // Read
    {
        SafeTensorsReader reader(path);
        REQUIRE(reader.size() == 1);
        REQUIRE(reader.has_tensor("weight"));
        REQUIRE_FALSE(reader.has_tensor("nonexistent"));

        auto names = reader.tensor_names();
        REQUIRE(names.size() == 1);
        REQUIRE(names[0] == "weight");

        const auto& info = reader.tensor_info("weight");
        REQUIRE(info.dtype == DataType::FP32);
        REQUIRE(info.shape == std::vector<std::int64_t>({3, 4}));
        REQUIRE(info.data_size == bytes.size());

        auto loaded = reader.read_tensor("weight");
        REQUIRE(loaded.size() == bytes.size());
        REQUIRE(loaded == bytes);

        // Verify float values
        std::vector<float> loaded_floats(12);
        std::memcpy(loaded_floats.data(), loaded.data(), loaded.size());
        for(int i = 0; i < 12; ++i)
        {
            REQUIRE(loaded_floats[i] == original[i]);
        }
    }

    remove_temp(path);
}

TEST_CASE("SafeTensors multiple tensors round-trip", "[io][safetensors]")
{
    const std::string path = temp_path("multi");

    // Create weight: 3x4 FP32
    std::vector<float> w(12, 1.0f);
    std::vector<std::uint8_t> w_bytes(w.size() * sizeof(float));
    std::memcpy(w_bytes.data(), w.data(), w_bytes.size());

    // Create bias: 4 FP32
    std::vector<float> b = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<std::uint8_t> b_bytes(b.size() * sizeof(float));
    std::memcpy(b_bytes.data(), b.data(), b_bytes.size());

    // Write
    {
        SafeTensorsWriter writer;
        writer.add_tensor("fc.weight", DataType::FP32, {3, 4}, w_bytes);
        writer.add_tensor("fc.bias", DataType::FP32, {4}, b_bytes);
        REQUIRE(writer.size() == 2);
        writer.write(path);
    }

    // Read
    {
        SafeTensorsReader reader(path);
        REQUIRE(reader.size() == 2);
        REQUIRE(reader.has_tensor("fc.weight"));
        REQUIRE(reader.has_tensor("fc.bias"));

        auto w_loaded = reader.read_tensor("fc.weight");
        REQUIRE(w_loaded == w_bytes);

        auto b_loaded = reader.read_tensor("fc.bias");
        REQUIRE(b_loaded == b_bytes);

        // Info checks
        REQUIRE(reader.tensor_info("fc.weight").shape ==
                std::vector<std::int64_t>({3, 4}));
        REQUIRE(reader.tensor_info("fc.bias").shape ==
                std::vector<std::int64_t>({4}));
    }

    remove_temp(path);
}

TEST_CASE("SafeTensors FP64 tensor round-trip", "[io][safetensors]")
{
    const std::string path = temp_path("fp64");

    std::vector<double> original = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<std::uint8_t> bytes(original.size() * sizeof(double));
    std::memcpy(bytes.data(), original.data(), bytes.size());

    {
        SafeTensorsWriter writer;
        writer.add_tensor("tensor", DataType::FP64, {2, 3}, bytes);
        writer.write(path);
    }

    {
        SafeTensorsReader reader(path);
        const auto& info = reader.tensor_info("tensor");
        REQUIRE(info.dtype == DataType::FP64);
        REQUIRE(info.shape == std::vector<std::int64_t>({2, 3}));

        auto loaded = reader.read_tensor("tensor");
        REQUIRE(loaded == bytes);
    }

    remove_temp(path);
}

TEST_CASE("SafeTensors INT64 tensor round-trip", "[io][safetensors]")
{
    const std::string path = temp_path("int64");

    std::vector<std::int64_t> original = {10, 20, 30};
    std::vector<std::uint8_t> bytes(original.size() * sizeof(std::int64_t));
    std::memcpy(bytes.data(), original.data(), bytes.size());

    {
        SafeTensorsWriter writer;
        writer.add_tensor("ids", DataType::INT64, {3}, bytes);
        writer.write(path);
    }

    {
        SafeTensorsReader reader(path);
        const auto& info = reader.tensor_info("ids");
        REQUIRE(info.dtype == DataType::INT64);
        auto loaded = reader.read_tensor("ids");
        REQUIRE(loaded == bytes);
    }

    remove_temp(path);
}

TEST_CASE("SafeTensors BOOL tensor round-trip", "[io][safetensors]")
{
    const std::string path = temp_path("bool");

    std::vector<std::uint8_t> original = {1, 0, 1, 1, 0};

    {
        SafeTensorsWriter writer;
        writer.add_tensor("mask", DataType::BOOL, {5}, original);
        writer.write(path);
    }

    {
        SafeTensorsReader reader(path);
        auto loaded = reader.read_tensor("mask");
        REQUIRE(loaded == original);
    }

    remove_temp(path);
}

TEST_CASE("SafeTensors scalar tensor (empty shape)", "[io][safetensors]")
{
    const std::string path = temp_path("scalar");

    float val = 3.14f;
    std::vector<std::uint8_t> bytes(sizeof(float));
    std::memcpy(bytes.data(), &val, sizeof(float));

    {
        SafeTensorsWriter writer;
        writer.add_tensor("scalar", DataType::FP32, {}, bytes);
        writer.write(path);
    }

    {
        SafeTensorsReader reader(path);
        auto loaded = reader.read_tensor("scalar");
        REQUIRE(loaded == bytes);

        float loaded_val = 0;
        std::memcpy(&loaded_val, loaded.data(), sizeof(float));
        REQUIRE(loaded_val == val);
    }

    remove_temp(path);
}

TEST_CASE("SafeTensors empty file (zero tensors)", "[io][safetensors]")
{
    const std::string path = temp_path("empty");

    {
        SafeTensorsWriter writer;
        REQUIRE(writer.size() == 0);
        writer.write(path);
    }

    {
        SafeTensorsReader reader(path);
        REQUIRE(reader.size() == 0);
        REQUIRE(reader.tensor_names().empty());
    }

    remove_temp(path);
}

// -------------------------------------------------------------------------
// Error handling tests
// -------------------------------------------------------------------------

TEST_CASE("SafeTensorsWriter rejects duplicate names", "[io][safetensors]")
{
    SafeTensorsWriter writer;
    std::vector<std::uint8_t> data(4 * sizeof(float), 0);
    writer.add_tensor("x", DataType::FP32, {4}, data);

    REQUIRE_THROWS_AS(
        writer.add_tensor("x", DataType::FP32, {4}, data),
        std::invalid_argument);
}

TEST_CASE("SafeTensorsWriter rejects data size mismatch", "[io][safetensors]")
{
    SafeTensorsWriter writer;
    std::vector<std::uint8_t> data(10, 0);

    REQUIRE_THROWS_AS(
        writer.add_tensor("x", DataType::FP32, {4}, data),
        std::invalid_argument);
}

TEST_CASE("SafeTensorsReader rejects nonexistent file", "[io][safetensors]")
{
    REQUIRE_THROWS_AS(
        SafeTensorsReader("/nonexistent/path.safetensors"),
        std::runtime_error);
}

TEST_CASE("SafeTensorsReader rejects missing tensor name", "[io][safetensors]")
{
    const std::string path = temp_path("missing_name");

    {
        SafeTensorsWriter writer;
        std::vector<std::uint8_t> data(sizeof(float), 0);
        writer.add_tensor("a", DataType::FP32, {1}, data);
        writer.write(path);
    }

    {
        SafeTensorsReader reader(path);
        REQUIRE_THROWS_AS(
            reader.tensor_info("nonexistent"),
            std::runtime_error);
        REQUIRE_THROWS_AS(
            reader.read_tensor("nonexistent"),
            std::runtime_error);
    }

    remove_temp(path);
}

// -------------------------------------------------------------------------
// Convenience free function tests
// -------------------------------------------------------------------------

TEST_CASE("save_tensor and load_tensor convenience functions", "[io][safetensors]")
{
    const std::string path = temp_path("convenience");

    std::vector<float> original = {1.0f, 2.0f, 3.0f};
    std::vector<std::uint8_t> bytes(original.size() * sizeof(float));
    std::memcpy(bytes.data(), original.data(), bytes.size());

    save_tensor(path, "vec", DataType::FP32, {3}, bytes);

    auto loaded = load_tensor(path, "vec");
    REQUIRE(loaded == bytes);

    remove_temp(path);
}

// -------------------------------------------------------------------------
// SafeTensorsWriter clear
// -------------------------------------------------------------------------

TEST_CASE("SafeTensorsWriter clear resets state", "[io][safetensors]")
{
    SafeTensorsWriter writer;
    std::vector<std::uint8_t> data(sizeof(float), 0);
    writer.add_tensor("a", DataType::FP32, {1}, data);
    REQUIRE(writer.size() == 1);

    writer.clear();
    REQUIRE(writer.size() == 0);

    writer.add_tensor("a", DataType::FP32, {1}, data);
    REQUIRE(writer.size() == 1);
}
