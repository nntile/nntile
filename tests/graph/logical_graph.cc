/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/logical_graph.cc
 * Tests for LogicalGraph class.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <algorithm>
#include <string>
#include <variant>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

// Include other NNTile headers
#include "nntile/graph.hh"

using namespace nntile;
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

TEST_CASE("LogicalGraph DataTypeHelpers", "[graph]")
{
    REQUIRE(dtype_to_string(DataType::FP32) == "FP32");
    REQUIRE(dtype_to_string(DataType::FP32_FAST_TF32) == "FP32_FAST_TF32");
    REQUIRE(dtype_to_string(DataType::FP32_FAST_FP16) == "FP32_FAST_FP16");
    REQUIRE(dtype_to_string(DataType::FP32_FAST_BF16) == "FP32_FAST_BF16");
    REQUIRE(dtype_to_string(DataType::FP64) == "FP64");
    REQUIRE(dtype_to_string(DataType::FP16) == "FP16");
    REQUIRE(dtype_to_string(DataType::BF16) == "BF16");
    REQUIRE(dtype_to_string(DataType::INT64) == "INT64");
    REQUIRE(dtype_to_string(DataType::INT32) == "INT32");
    REQUIRE(dtype_to_string(DataType::BOOL) == "BOOL");

    REQUIRE(dtype_size(DataType::BOOL) == 1);
    REQUIRE(dtype_size(DataType::FP16) == 2);
    REQUIRE(dtype_size(DataType::BF16) == 2);
    REQUIRE(dtype_size(DataType::FP32) == 4);
    REQUIRE(dtype_size(DataType::FP32_FAST_TF32) == 4);
    REQUIRE(dtype_size(DataType::FP32_FAST_FP16) == 4);
    REQUIRE(dtype_size(DataType::FP32_FAST_BF16) == 4);
    REQUIRE(dtype_size(DataType::INT32) == 4);
    REQUIRE(dtype_size(DataType::FP64) == 8);
    REQUIRE(dtype_size(DataType::INT64) == 8);

    REQUIRE_THROWS_AS(
        dtype_to_string(static_cast<DataType>(-1)),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        dtype_size(static_cast<DataType>(-1)),
        std::invalid_argument);
}

TEST_CASE("LogicalGraph OpTypeHelpers", "[graph]")
{
    REQUIRE(op_type_to_string(OpType::GEMM) == "GEMM");
    REQUIRE(op_type_to_string(OpType::GELU) == "GELU");
    REQUIRE(op_type_to_string(OpType::GELU_BACKWARD) == "GELU_BACKWARD");
    REQUIRE(op_type_to_string(OpType::ADD_FIBER) == "ADD_FIBER");
    REQUIRE(op_type_to_string(OpType::SUM_FIBER) == "SUM_FIBER");
    REQUIRE(op_type_to_string(OpType::CLEAR) == "CLEAR");

    REQUIRE_THROWS_AS(
        op_type_to_string(static_cast<OpType>(-1)),
        std::invalid_argument);
}

TEST_CASE("LogicalGraph TensorNodeDimAndSize", "[graph]")
{
    LogicalGraph g("test");
    auto& x = g.tensor({2, 3, 4}, "x", DataType::FP16);

    REQUIRE(x.dim(0) == 2);
    REQUIRE(x.dim(1) == 3);
    REQUIRE(x.dim(-1) == 4);
    REQUIRE(x.dim(-3) == 2);
    REQUIRE(x.nelems() == 24);
    REQUIRE(x.size_bytes() == 24 * dtype_size(DataType::FP16));

    REQUIRE_THROWS_AS(x.dim(3), std::out_of_range);
    REQUIRE_THROWS_AS(x.dim(-4), std::out_of_range);
}

TEST_CASE("LogicalGraph TensorNodeCompatibilityAndString", "[graph]")
{
    LogicalGraph g("test");
    auto& a = g.tensor({2}, "a", DataType::FP32);
    auto& b = g.tensor({2}, "b", DataType::FP32);
    auto& c = g.tensor({2}, "c", DataType::INT64);

    REQUIRE(a.is_compatible(b));
    REQUIRE_FALSE(a.is_compatible(c));

    auto text = a.to_string();
    REQUIRE(text.find("LogicalGraph::TensorNode") != std::string::npos);
    REQUIRE(text.find("name='a'") != std::string::npos);
    REQUIRE(text.find("dtype=FP32") != std::string::npos);
}

TEST_CASE("LogicalGraph TensorInvalidShape", "[graph]")
{
    LogicalGraph g("test");
    REQUIRE_THROWS_AS(g.tensor({0}, "x"), std::invalid_argument);
    REQUIRE_THROWS_AS(g.tensor({-1, 2}, "y"), std::invalid_argument);
}

TEST_CASE("LogicalGraph AddOpValidations", "[graph]")
{
    LogicalGraph g1("g1");
    LogicalGraph g2("g2");

    auto& a = g1.tensor({2, 2}, "a", DataType::FP32);
    auto& b = g1.tensor({2, 2}, "b", DataType::FP32);
    auto& c = g1.tensor({2, 2}, "c", DataType::FP32);

    auto& foreign = g2.tensor({2, 2}, "foreign", DataType::FP32);
    auto& foreign_out = g2.tensor({2, 2}, "foreign_out", DataType::FP32);

    REQUIRE_THROWS_AS(
        g1.add_op(OpType::GEMM, GemmAttrs{}, {&a, &foreign}, {&c}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        g1.add_op(OpType::CLEAR, ClearAttrs{}, {}, {&foreign_out}),
        std::invalid_argument);

    g1.add_op(OpType::GEMM, GemmAttrs{}, {&a, &b}, {&c}, "gemm_op");
    REQUIRE(g1.num_ops() == 1);
    REQUIRE(c.has_producer());
    REQUIRE(c.producer()->type() == OpType::GEMM);
}

TEST_CASE("LogicalGraph ToStringContainsDetails", "[graph]")
{
    LogicalGraph g("test");
    auto& a = g.tensor({2, 3}, "a", DataType::FP32);
    auto& b = g.tensor({3, 4}, "b", DataType::FP32);
    auto& c = gemm(a, b, "c");

    auto graph_text = g.to_string();
    REQUIRE(graph_text.find("LogicalGraph(name='test'") != std::string::npos);
    REQUIRE(graph_text.find("tensors=3") != std::string::npos);
    REQUIRE(graph_text.find("ops=1") != std::string::npos);

    auto op_text = g.ops().front()->to_string();
    REQUIRE(op_text.find("GEMM") != std::string::npos);
    REQUIRE(op_text.find("inputs=[a, b]") != std::string::npos);
    REQUIRE(op_text.find("outputs=[c]") != std::string::npos);
}
