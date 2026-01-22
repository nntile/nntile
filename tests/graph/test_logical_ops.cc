/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/test_logical_ops.cc
 * Tests for logical graph operations (excluding GEMM).
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <string>
#include <variant>

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
