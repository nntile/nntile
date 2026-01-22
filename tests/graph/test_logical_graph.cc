/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/test_logical_graph.cc
 * Tests for LogicalGraph class.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <algorithm>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

// Include other NNTile headers
#include "nntile/graph.hh"

using namespace nntile::graph;

TEST_CASE("LogicalGraph CreateTensor", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);

    REQUIRE(x.name() == "x");
    REQUIRE(x.shape().size() == 2);
    REQUIRE(x.shape()[0] == 32);
    REQUIRE(x.shape()[1] == 768);
    REQUIRE(x.dtype() == DataType::FP32);
    REQUIRE_FALSE(x.has_producer());
}

TEST_CASE("LogicalGraph Gemm", "[graph]")
{
    LogicalGraph g("test");

    auto& a = g.tensor({32, 768}, "a", DataType::FP32);
    auto& b = g.tensor({768, 256}, "b", DataType::FP32);
    auto& c = gemm(a, b, "c");

    REQUIRE(c.shape()[0] == 32);
    REQUIRE(c.shape()[1] == 256);
    REQUIRE(c.has_producer());
    REQUIRE(c.producer()->type() == OpType::GEMM);
}

TEST_CASE("LogicalGraph GemmTranspose", "[graph]")
{
    LogicalGraph g("test");

    auto& a = g.tensor(
        {768, 32},
        "a",
        DataType::FP32);  // Will be transposed
    auto& b = g.tensor({768, 256}, "b", DataType::FP32);
    auto& c = gemm(
        a,
        b,
        "c",
        1.0,
        /*trans_a=*/true,
        /*trans_b=*/false);

    REQUIRE(c.shape()[0] == 32);   // M from A^T
    REQUIRE(c.shape()[1] == 256);  // N from B
}

TEST_CASE("LogicalGraph Gelu", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = gelu(x, "y");

    REQUIRE(y.shape() == x.shape());
    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::GELU);
}

TEST_CASE("LogicalGraph Clear", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({16, 16}, "x", DataType::FP32);
    clear(x);

    REQUIRE(x.has_producer());
    REQUIRE(x.producer()->type() == OpType::CLEAR);
    REQUIRE(x.producer()->inputs().size() == 0);
    REQUIRE(x.producer()->outputs().size() == 1);
}

TEST_CASE("LogicalGraph Chain", "[graph]")
{
    LogicalGraph g("mlp");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& w1 = g.tensor({768, 3072}, "w1", DataType::FP32);
    auto& w2 = g.tensor({3072, 768}, "w2", DataType::FP32);

    auto& h = gemm(x, w1, "h");
    auto& a = gelu(h, "a");
    auto& y = gemm(a, w2, "y");

    REQUIRE(g.num_tensors() == 6);  // x, w1, w2, h, a, y
    REQUIRE(g.num_ops() == 3);      // gemm, gelu, gemm
}

TEST_CASE("LogicalGraph TensorNameUniqueness", "[graph]")
{
    LogicalGraph g("test");

    g.tensor({10}, "x", DataType::FP32);
    REQUIRE_THROWS_AS(
        g.tensor({10}, "x", DataType::FP32),
        std::invalid_argument);
}

TEST_CASE("LogicalGraph GetTensor", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({10}, "x", DataType::FP32);
    auto& y = g.tensor({20}, "y", DataType::FP32);

    REQUIRE(g.get_tensor("x") == &x);
    REQUIRE(g.get_tensor("y") == &y);
    REQUIRE(g.get_tensor("z") == nullptr);
}

TEST_CASE("LogicalGraph TensorNames", "[graph]")
{
    LogicalGraph g("test");

    g.tensor({10}, "x", DataType::FP32);
    g.tensor({20}, "y", DataType::FP32);

    auto names = g.tensor_names();
    REQUIRE(names.size() == 2);
    REQUIRE(std::find(names.begin(), names.end(), "x") != names.end());
    REQUIRE(std::find(names.begin(), names.end(), "y") != names.end());
}
