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

#include <nntile/graph/graph.hh>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>

using namespace nntile::graph;

TEST_CASE("CompiledGraph SimpleMatmul", "[graph]") {
    LogicalGraph g("test");

    auto& a = g.tensor(TensorSpec({2, 3}, DataType::FP32), "a");
    auto& b = g.tensor(TensorSpec({3, 4}, DataType::FP32), "b");
    auto& c = g.matmul(a, b, "c");
    g.mark_output("c");

    auto compiled = CompiledGraph::compile(g);

    // A = [[1,2,3], [4,5,6]]
    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    // B = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    std::vector<float> b_data = {1,2,3,4, 5,6,7,8, 9,10,11,12};

    compiled.bind_data("a", a_data);
    compiled.bind_data("b", b_data);

    compiled.execute();
    compiled.wait();

    auto c_data = compiled.get_output<float>("c");

    // C = A @ B = [[38,44,50,56], [83,98,113,128]]
    REQUIRE(c_data.size() == 8);
    REQUIRE(c_data[0] == 38);
    REQUIRE(c_data[1] == 44);
    REQUIRE(c_data[2] == 50);
    REQUIRE(c_data[3] == 56);
    REQUIRE(c_data[4] == 83);
    REQUIRE(c_data[5] == 98);
    REQUIRE(c_data[6] == 113);
    REQUIRE(c_data[7] == 128);
}

TEST_CASE("CompiledGraph GeluActivation", "[graph]") {
    LogicalGraph g("test");

    auto& x = g.tensor(TensorSpec({4}, DataType::FP32), "x");
    auto& y = g.gelu(x, "y");
    g.mark_output("y");

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

TEST_CASE("CompiledGraph MLP", "[graph]") {
    LogicalGraph g("mlp");

    // x: [2, 4], w1: [4, 8], w2: [8, 4]
    auto& x = g.tensor(TensorSpec({2, 4}, DataType::FP32), "x");
    auto& w1 = g.tensor(TensorSpec({4, 8}, DataType::FP32), "w1");
    auto& w2 = g.tensor(TensorSpec({8, 4}, DataType::FP32), "w2");

    auto& h = g.matmul(x, w1, "h");
    auto& a = g.gelu(h, "a");
    auto& y = g.matmul(a, w2, "y");
    g.mark_output("y");

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
    for (float v : y_data) {
        REQUIRE(v > 0.0f);
        REQUIRE(v < 10.0f);
    }
}

TEST_CASE("CompiledGraph DataTypeMismatch", "[graph]") {
    LogicalGraph g("test");

    auto& a = g.tensor(TensorSpec({2, 2}, DataType::FP32), "a");
    auto& b = g.tensor(TensorSpec({2, 2}, DataType::FP32), "b");
    auto& c = g.matmul(a, b, "c");
    g.mark_output("c");

    auto compiled = CompiledGraph::compile(g);

    // Wrong size
    std::vector<float> wrong_data(1, 1.0f);
    REQUIRE_THROWS_AS(compiled.bind_data("a", wrong_data), std::runtime_error);
}

TEST_CASE("CompiledGraph UnsupportedDataType", "[graph]") {
    LogicalGraph g("test");

    auto& x = g.tensor(TensorSpec({2}, DataType::INT64), "x");
    auto& y = g.gelu(x, "y");
    g.mark_output("y");

    auto compiled = CompiledGraph::compile(g);

    std::vector<long long> x_data(2, 1);
    compiled.bind_data("x", x_data);

    // Should throw when executing unsupported operation
    REQUIRE_THROWS_AS(compiled.execute(), std::runtime_error);
}