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

#include <nntile/graph/graph.hh>
#include <gtest/gtest.h>

using namespace nntile::graph;

TEST(LogicalGraph, CreateTensor) {
    LogicalGraph g("test");

    auto& x = g.tensor(TensorSpec({32, 768}, DataType::FP32), "x");

    EXPECT_EQ(x.name(), "x");
    EXPECT_EQ(x.shape().size(), 2);
    EXPECT_EQ(x.shape()[0], 32);
    EXPECT_EQ(x.shape()[1], 768);
    EXPECT_EQ(x.dtype(), DataType::FP32);
    EXPECT_FALSE(x.has_producer());
}

TEST(LogicalGraph, Matmul) {
    LogicalGraph g("test");

    auto& a = g.tensor(TensorSpec({32, 768}, DataType::FP32), "a");
    auto& b = g.tensor(TensorSpec({768, 256}, DataType::FP32), "b");
    auto& c = g.matmul(a, b, "c");

    EXPECT_EQ(c.shape()[0], 32);
    EXPECT_EQ(c.shape()[1], 256);
    EXPECT_TRUE(c.has_producer());
    EXPECT_EQ(c.producer()->type(), OpType::MATMUL);
}

TEST(LogicalGraph, MatmulTranspose) {
    LogicalGraph g("test");

    auto& a = g.tensor(TensorSpec({768, 32}, DataType::FP32), "a");  // Will be transposed
    auto& b = g.tensor(TensorSpec({768, 256}, DataType::FP32), "b");
    auto& c = g.matmul(a, b, "c", /*trans_a=*/true, /*trans_b=*/false);

    EXPECT_EQ(c.shape()[0], 32);   // M from A^T
    EXPECT_EQ(c.shape()[1], 256);  // N from B
}

TEST(LogicalGraph, Gelu) {
    LogicalGraph g("test");

    auto& x = g.tensor(TensorSpec({32, 768}, DataType::FP32), "x");
    auto& y = g.gelu(x, "y");

    EXPECT_EQ(y.shape(), x.shape());
    EXPECT_TRUE(y.has_producer());
    EXPECT_EQ(y.producer()->type(), OpType::GELU);
}

TEST(LogicalGraph, Chain) {
    LogicalGraph g("mlp");

    auto& x = g.tensor(TensorSpec({32, 768}, DataType::FP32), "x");
    auto& w1 = g.tensor(TensorSpec({768, 3072}, DataType::FP32), "w1");
    auto& w2 = g.tensor(TensorSpec({3072, 768}, DataType::FP32), "w2");

    auto& h = g.matmul(x, w1, "h");
    auto& a = g.gelu(h, "a");
    auto& y = g.matmul(a, w2, "y");

    g.mark_output("y");

    EXPECT_EQ(g.num_tensors(), 6);  // x, w1, w2, h, a, y
    EXPECT_EQ(g.num_ops(), 3);      // matmul, gelu, matmul
    EXPECT_TRUE(g.is_output("y"));
}

TEST(LogicalGraph, TensorNameUniqueness) {
    LogicalGraph g("test");

    g.tensor(TensorSpec({10}, DataType::FP32), "x");
    EXPECT_THROW(g.tensor(TensorSpec({10}, DataType::FP32), "x"), std::invalid_argument);
}

TEST(LogicalGraph, GetTensor) {
    LogicalGraph g("test");

    auto& x = g.tensor(TensorSpec({10}, DataType::FP32), "x");
    auto& y = g.tensor(TensorSpec({20}, DataType::FP32), "y");

    EXPECT_EQ(g.get_tensor("x"), &x);
    EXPECT_EQ(g.get_tensor("y"), &y);
    EXPECT_EQ(g.get_tensor("z"), nullptr);
}

TEST(LogicalGraph, TensorNames) {
    LogicalGraph g("test");

    g.tensor(TensorSpec({10}, DataType::FP32), "x");
    g.tensor(TensorSpec({20}, DataType::FP32), "y");

    auto names = g.tensor_names();
    EXPECT_EQ(names.size(), 2);
    EXPECT_NE(std::find(names.begin(), names.end(), "x"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "y"), names.end());
}