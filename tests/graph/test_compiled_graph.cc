/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/test_compiled_graph.cc
 * Tests for CompiledGraph class.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>

#include "nntile/context.hh"
#include "nntile/graph.hh"

using namespace nntile::graph;

// Fixture to initialize NNTile context for graph tests
class GraphTestFixture
{
protected:
    nntile::Context context;
public:
    GraphTestFixture():
        context(
            1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0
        )
    {}
};

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SimpleGemm",
    "[graph]"
)
{
    LogicalGraph g("test");

    auto& a = g.tensor(TensorSpec({2, 3}, DataType::FP32), "a");
    auto& b = g.tensor(TensorSpec({3, 4}, DataType::FP32), "b");
    auto& c = gemm(a, b, "c");

    auto compiled = CompiledGraph::compile(g);

    // A = [[1,2,3], [4,5,6]]
    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    // B = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    std::vector<float> b_data = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    };

    compiled.bind_data("a", a_data);
    compiled.bind_data("b", b_data);

    compiled.execute();
    compiled.wait();

    auto c_data = compiled.get_output<float>("c");

    // C = A @ B in column-major storage
    // A = [[1,3,5], [2,4,6]], B = [[1,4,7,10], [2,5,8,11], [3,6,9,12]]
    // C = [[22,49,76,103], [28,64,100,136]]
    REQUIRE(c_data.size() == 8);
    REQUIRE(c_data[0] == 22);
    REQUIRE(c_data[1] == 28);
    REQUIRE(c_data[2] == 49);
    REQUIRE(c_data[3] == 64);
    REQUIRE(c_data[4] == 76);
    REQUIRE(c_data[5] == 100);
    REQUIRE(c_data[6] == 103);
    REQUIRE(c_data[7] == 136);
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph GeluActivation",
    "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor(TensorSpec({4}, DataType::FP32), "x");
    auto& y = gelu(x, "y");

    auto compiled = CompiledGraph::compile(g);

    std::vector<float> x_data = {-1.0f, 0.0f, 1.0f, 2.0f};
    compiled.bind_data("x", x_data);

    compiled.execute();
    compiled.wait();

    auto y_data = compiled.get_output<float>("y");

    // GELU(-1) ≈ -0.159, GELU(0) = 0, GELU(1) ≈ 0.841, GELU(2) ≈ 1.955
    REQUIRE(y_data[0] == Catch::Approx(-0.159f).epsilon(0.01));
    REQUIRE(y_data[1] == Catch::Approx(0.0f).epsilon(0.01));
    REQUIRE(y_data[2] == Catch::Approx(0.841f).epsilon(0.01));
    REQUIRE(y_data[3] == Catch::Approx(1.955f).epsilon(0.01));
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph MLP",
    "[graph]")
{
    LogicalGraph g("mlp");

    // x: [2, 4], w1: [4, 8], w2: [8, 4]
    auto& x = g.tensor(TensorSpec({2, 4}, DataType::FP32), "x");
    auto& w1 = g.tensor(TensorSpec({4, 8}, DataType::FP32), "w1");
    auto& w2 = g.tensor(TensorSpec({8, 4}, DataType::FP32), "w2");

    auto& h = gemm(x, w1, "h");
    auto& a = gelu(h, "a");
    auto& y = gemm(a, w2, "y");

    auto compiled = CompiledGraph::compile(g);

    // Initialize with simple values
    std::vector<float> x_data(8, 1.0f);
    std::vector<float> w1_data(32, 0.1f);
    std::vector<float> w2_data(32, 0.1f);

    compiled.bind_data("x", x_data);
    compiled.bind_data("w1", w1_data);
    compiled.bind_data("w2", w2_data);

    compiled.execute();
    compiled.wait();

    auto y_data = compiled.get_output<float>("y");

    REQUIRE(y_data.size() == 8);  // [2, 4]
    // Values should be non-zero and reasonable
    for(float v : y_data)
    {
        REQUIRE(v > 0.0f);
        REQUIRE(v < 10.0f);
    }
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph DataTypeMismatch",
    "[graph]")
{
    LogicalGraph g("test");

    auto& a = g.tensor(TensorSpec({2, 2}, DataType::FP32), "a");
    auto& b = g.tensor(TensorSpec({2, 2}, DataType::FP32), "b");
    auto& c = gemm(a, b, "c");

    auto compiled = CompiledGraph::compile(g);

    // Wrong size
    std::vector<float> wrong_data(1, 1.0f);
    REQUIRE_THROWS_AS(compiled.bind_data("a", wrong_data), std::runtime_error);
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph UnsupportedDataType",
    "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor(TensorSpec({2}, DataType::INT64), "x");
    auto& y = gelu(x, "y");

    // Should throw during compilation for unsupported data type
    REQUIRE_THROWS_AS(CompiledGraph::compile(g), std::runtime_error);
}
