/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/logical_graph_ops.cc
 * Tests for logical graph operations.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <string>
#include <variant>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

// Include other NNTile headers
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

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

TEST_CASE("LogicalOp GeluCreatesOutputWithAttrs", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({2, 3}, "x", DataType::FP32);
    auto& y = gelu(x, "y");

    REQUIRE(y.shape() == x.shape());
    REQUIRE(y.dtype() == x.dtype());
    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::GELU);
    REQUIRE(std::holds_alternative<GeluAttrs>(y.producer()->attrs()));
}

TEST_CASE("LogicalOp GeluBackward", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({2, 3}, "x", DataType::FP32);
    auto& dy = g.tensor({2, 3}, "dy", DataType::FP32);
    auto& dx = g.tensor({2, 3}, "dx", DataType::FP32);

    gelu_backward(x, dy, dx);

    REQUIRE(dx.has_producer());
    REQUIRE(dx.producer()->type() == OpType::GELU_BACKWARD);
    REQUIRE(dx.producer()->inputs().size() == 3);
    REQUIRE(dx.producer()->outputs().size() == 1);
    REQUIRE(dx.producer()->output() == &dx);
    REQUIRE(std::holds_alternative<GeluBackwardAttrs>(dx.producer()->attrs()));
}

TEST_CASE("LogicalOp GeluBackwardDifferentGraph", "[graph]")
{
    LogicalGraph g("test");
    LogicalGraph other("other");

    auto& x = g.tensor({2, 3}, "x", DataType::FP32);
    auto& dy = other.tensor({2, 3}, "dy", DataType::FP32);
    auto& dx = g.tensor({2, 3}, "dx", DataType::FP32);

    REQUIRE_THROWS_AS(gelu_backward(x, dy, dx), std::invalid_argument);
}

TEST_CASE("LogicalOp ClearAttributes", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({2, 2}, "x", DataType::FP32);
    clear(x);

    REQUIRE(x.has_producer());
    REQUIRE(x.producer()->type() == OpType::CLEAR);
    REQUIRE(x.producer()->inputs().empty());
    REQUIRE(x.producer()->outputs().size() == 1);
    REQUIRE(x.producer()->output() == &x);
    REQUIRE(std::holds_alternative<ClearAttrs>(x.producer()->attrs()));
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

TEST_CASE("LogicalOp GemmCreatesOutputWithAttrs", "[graph]")
{
    LogicalGraph g("test");

    auto& a = g.tensor({2, 3, 4}, "a", DataType::FP32);
    auto& b = g.tensor({3, 5, 4}, "b", DataType::FP32);
    auto& c = gemm(a, b, "c", 2.0f, false, false, 1, 1);

    REQUIRE(c.shape() == std::vector<Index>({2, 5, 4}));
    REQUIRE(c.has_producer());
    REQUIRE(c.producer()->type() == OpType::GEMM);
    REQUIRE(std::holds_alternative<GemmAttrs>(c.producer()->attrs()));

    auto attrs = std::get<GemmAttrs>(c.producer()->attrs());
    REQUIRE(attrs.alpha == 2.0f);
    REQUIRE(attrs.beta == 0.0f);
    REQUIRE_FALSE(attrs.trans_a);
    REQUIRE_FALSE(attrs.trans_b);
    REQUIRE(attrs.ndim == 1);
    REQUIRE(attrs.batch_ndim == 1);
}

TEST_CASE("LogicalOp GemmTransposedOutput", "[graph]")
{
    LogicalGraph g("test");

    auto& a = g.tensor({3, 2}, "a", DataType::FP32);
    auto& b = g.tensor({3, 5}, "b", DataType::FP32);
    auto& c = gemm(a, b, "c", 1.0f, true, false, 1, 0);

    REQUIRE(c.shape() == std::vector<Index>({2, 5}));
}

TEST_CASE("LogicalOp GemmAccumulation", "[graph]")
{
    LogicalGraph g("test");

    auto& a = g.tensor({2, 3}, "a", DataType::FP32);
    auto& b = g.tensor({3, 4}, "b", DataType::FP32);
    auto& c = g.tensor({2, 4}, "c", DataType::FP32);

    auto& out = gemm(a, b, c, 2.0f, 3.0f);

    REQUIRE(&out == &c);
    REQUIRE(c.has_producer());
    REQUIRE(c.producer()->type() == OpType::GEMM);
    REQUIRE(c.producer()->inputs().size() == 3);
    REQUIRE(c.producer()->outputs().size() == 1);
    REQUIRE(c.producer()->output() == &c);
    REQUIRE(std::holds_alternative<GemmAttrs>(c.producer()->attrs()));

    auto attrs = std::get<GemmAttrs>(c.producer()->attrs());
    REQUIRE(attrs.alpha == 2.0f);
    REQUIRE(attrs.beta == 3.0f);
}

TEST_CASE("LogicalOp GemmValidations", "[graph]")
{
    LogicalGraph g("test");
    LogicalGraph other("other");

    auto& a = g.tensor({2, 3}, "a", DataType::FP32);
    auto& b = g.tensor({3, 4}, "b", DataType::FP64);
    auto& c = g.tensor({2, 4}, "c", DataType::FP32);

    REQUIRE_THROWS_AS(gemm(a, b, "c_out"), std::invalid_argument);

    auto& other_b = other.tensor({3, 4}, "b_other", DataType::FP32);
    REQUIRE_THROWS_AS(gemm(a, other_b, "c_out"), std::invalid_argument);

    REQUIRE_THROWS_AS(
        gemm(a, a, "c_out", 1.0f, false, false, 0, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gemm(a, a, "c_out", 1.0f, false, false, 1, static_cast<Index>(-1)),
        std::invalid_argument);

    auto& b_bad = g.tensor({4, 4}, "b_bad", DataType::FP32);
    REQUIRE_THROWS_AS(gemm(a, b_bad, "c_bad"), std::invalid_argument);

    auto& a_batch = g.tensor({2, 3, 4}, "a_batch", DataType::FP32);
    auto& b_batch = g.tensor({3, 5, 6}, "b_batch", DataType::FP32);
    REQUIRE_THROWS_AS(
        gemm(a_batch, b_batch, "c_batch", 1.0f, false, false, 1, 1),
        std::invalid_argument);

    auto& c_bad = g.tensor({2, 5}, "c_bad", DataType::FP32);
    REQUIRE_THROWS_AS(gemm(a, a, c_bad), std::invalid_argument);
    REQUIRE_THROWS_AS(gemm(a, b, c), std::invalid_argument);
}
