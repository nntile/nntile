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

    gemm(a, b, c, 2.0f, 3.0f);

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

TEST_CASE("LogicalGraph Add", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = g.tensor({32, 768}, "y", DataType::FP32);
    auto& z = add(x, y, "z");

    REQUIRE(z.shape() == x.shape());
    REQUIRE(z.has_producer());
    REQUIRE(z.producer()->type() == OpType::ADD);
    REQUIRE(std::holds_alternative<BinaryOpAttrs>(z.producer()->attrs()));

    auto attrs = std::get<BinaryOpAttrs>(z.producer()->attrs());
    REQUIRE(attrs.alpha == 1.0);
    REQUIRE(attrs.beta == 1.0);
}

TEST_CASE("LogicalGraph AddInplace", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({16, 16}, "x", DataType::FP32);
    auto& y = g.tensor({16, 16}, "y", DataType::FP32);

    add_inplace(x, y);

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::ADD_INPLACE);
    REQUIRE(y.producer()->inputs().size() == 2);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph Sum", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = g.tensor({1}, "y", DataType::FP32);

    sum(x, y);

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::SUM);
    REQUIRE(std::holds_alternative<TotalSumAttrs>(y.producer()->attrs()));

    auto attrs = std::get<TotalSumAttrs>(y.producer()->attrs());
    REQUIRE(attrs.alpha == 1.0);
    REQUIRE(attrs.beta == 0.0);
}

TEST_CASE("LogicalGraph Embedding", "[graph]")
{
    LogicalGraph g("test");

    auto& index = g.tensor({32, 128}, "index", DataType::INT64);
    auto& vocab = g.tensor({768, 50000}, "vocab", DataType::FP32);  // [embed_dim, vocab_size]
    auto& embed = embedding(index, vocab, "embed");

    REQUIRE(embed.shape() == std::vector<Index>({768, 32, 128}));  // embed_dim inserted at axis 0
    REQUIRE(embed.has_producer());
    REQUIRE(embed.producer()->type() == OpType::EMBEDDING);
    REQUIRE(std::holds_alternative<EmbeddingAttrs>(embed.producer()->attrs()));

    auto attrs = std::get<EmbeddingAttrs>(embed.producer()->attrs());
    REQUIRE(attrs.axis == 0);
}

TEST_CASE("LogicalGraph EmbeddingBackward", "[graph]")
{
    LogicalGraph g("test");

    auto& embed = g.tensor({768, 32, 128}, "embed", DataType::FP32);  // [embed_dim, ...]
    auto& index = g.tensor({32, 128}, "index", DataType::INT64);
    auto& vocab = g.tensor({768, 50000}, "vocab", DataType::FP32);    // [embed_dim, vocab_size]

    embedding_backward(embed, index, vocab);

    REQUIRE(vocab.has_producer());
    REQUIRE(vocab.producer()->type() == OpType::EMBEDDING_BACKWARD);
    REQUIRE(vocab.producer()->inputs().size() == 3);
    REQUIRE(vocab.producer()->outputs().size() == 1);
    REQUIRE(vocab.producer()->output() == &vocab);
}

// Tests for newly implemented element-wise operations
TEST_CASE("LogicalGraph Hypot", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = g.tensor({32, 768}, "y", DataType::FP32);
    auto& z = hypot(x, y, "z", 2.0f, 3.0f);

    REQUIRE(z.shape() == x.shape());
    REQUIRE(z.has_producer());
    REQUIRE(z.producer()->type() == OpType::HYPOT);
    REQUIRE(std::holds_alternative<BinaryOpAttrs>(z.producer()->attrs()));

    auto attrs = std::get<BinaryOpAttrs>(z.producer()->attrs());
    REQUIRE(attrs.alpha == 2.0f);
    REQUIRE(attrs.beta == 3.0f);
}

TEST_CASE("LogicalGraph HypotInplace", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({16, 16}, "x", DataType::FP32);
    auto& y = g.tensor({16, 16}, "y", DataType::FP32);

    hypot_inplace(x, y, 1.5f, 2.5f);

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::HYPOT_INPLACE);
    REQUIRE(y.producer()->inputs().size() == 2);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph Pow", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = pow(x, "y", 2.0f, 3.0f);

    REQUIRE(y.shape() == x.shape());
    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::POW);
    REQUIRE(std::holds_alternative<PowAttrs>(y.producer()->attrs()));

    auto attrs = std::get<PowAttrs>(y.producer()->attrs());
    REQUIRE(attrs.alpha == 2.0f);
    REQUIRE(attrs.exponent == 3.0f);
}

TEST_CASE("LogicalGraph PowInplace", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({16, 16}, "x", DataType::FP32);
    pow_inplace(x, 1.5f, 2.0f);

    REQUIRE(x.has_producer());
    REQUIRE(x.producer()->type() == OpType::POW_INPLACE);
    REQUIRE(x.producer()->inputs().size() == 1);
    REQUIRE(x.producer()->outputs().size() == 1);
    REQUIRE(x.producer()->output() == &x);
}

TEST_CASE("LogicalGraph LogScalar", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({1}, "x", DataType::FP32);
    log_scalar(x, "test_value");

    REQUIRE(x.has_producer());
    REQUIRE(x.producer()->type() == OpType::LOG_SCALAR);
    REQUIRE(x.producer()->inputs().size() == 1);
    REQUIRE(x.producer()->outputs().size() == 0);  // No output tensor
}

TEST_CASE("LogicalGraph MaskScalar", "[graph]")
{
    LogicalGraph g("test");

    auto& mask = g.tensor({32, 768}, "mask", DataType::BOOL);
    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    mask_scalar(mask, x, 0.5f, 0);

    REQUIRE(x.has_producer());
    REQUIRE(x.producer()->type() == OpType::MASK_SCALAR);
    REQUIRE(x.producer()->inputs().size() == 2);
    REQUIRE(x.producer()->outputs().size() == 1);
    REQUIRE(x.producer()->output() == &x);
}

TEST_CASE("LogicalGraph HypotScalarInverse", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({16, 16}, "x", DataType::FP32);
    hypot_scalar_inverse(x, 1e-6f, 1.0f);

    REQUIRE(x.has_producer());
    REQUIRE(x.producer()->type() == OpType::HYPOT_SCALAR_INVERSE);
    REQUIRE(x.producer()->inputs().size() == 1);
    REQUIRE(x.producer()->outputs().size() == 1);
    REQUIRE(x.producer()->output() == &x);
}

TEST_CASE("LogicalGraph SubtractIndexedOutputs", "[graph]")
{
    LogicalGraph g("test");

    auto& labels = g.tensor({32, 128}, "labels", DataType::INT64);
    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    subtract_indexed_outputs(labels, x, 0.1f, -1);

    REQUIRE(x.has_producer());
    REQUIRE(x.producer()->type() == OpType::SUBTRACT_INDEXED_OUTPUTS);
    REQUIRE(x.producer()->inputs().size() == 2);
    REQUIRE(x.producer()->outputs().size() == 1);
    REQUIRE(x.producer()->output() == &x);
}

// Tests for reduction operations
TEST_CASE("LogicalGraph SumSlice", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = g.tensor({32, 1}, "y", DataType::FP32);
    sum_slice(x, y, 1, 0, 2.0f, 0.5f);  // Sum along axis 1

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::SUM_SLICE);
    REQUIRE(y.producer()->inputs().size() == 2);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph Norm", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = g.tensor({1}, "y", DataType::FP32);
    norm(x, y, 1.0f, 0.0f);

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::NORM);
    REQUIRE(y.producer()->inputs().size() == 2);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph Logsumexp", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = g.tensor({32, 1}, "y", DataType::FP32);
    logsumexp(x, y, 1);  // Along axis 1

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::LOGSUMEXP);
    REQUIRE(y.producer()->inputs().size() == 1);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph Maxsumexp", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = g.tensor({32, 1}, "y", DataType::FP32);
    maxsumexp(x, y, 1, 0);  // Along axis 1

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::MAXSUMEXP);
    REQUIRE(y.producer()->inputs().size() == 1);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph SumprodFiber", "[graph]")
{
    LogicalGraph g("test");

    auto& x1 = g.tensor({32, 768}, "x1", DataType::FP32);
    auto& x2 = g.tensor({32, 768}, "x2", DataType::FP32);
    auto& y = g.tensor({32, 1}, "y", DataType::FP32);
    sumprod_fiber(x1, x2, y, 1, 0, 1.0f, 0.0f);  // Sum along axis 1

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::SUMPROD_FIBER);
    REQUIRE(y.producer()->inputs().size() == 3);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph SumprodSlice", "[graph]")
{
    LogicalGraph g("test");

    auto& x1 = g.tensor({32, 768}, "x1", DataType::FP32);
    auto& x2 = g.tensor({32, 768}, "x2", DataType::FP32);
    auto& y = g.tensor({32, 1}, "y", DataType::FP32);
    sumprod_slice(x1, x2, y, 1, 0, 1.0f, 0.0f);  // Sum along axis 1

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::SUMPROD_SLICE);
    REQUIRE(y.producer()->inputs().size() == 3);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph NormFiber", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = g.tensor({32, 1}, "y", DataType::FP32);
    norm_fiber(x, y, 1, 0, 0, 1.0f, 0.0f);  // Along axis 1

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::NORM_FIBER);
    REQUIRE(y.producer()->inputs().size() == 2);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph NormSlice", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = g.tensor({32, 1}, "y", DataType::FP32);
    norm_slice(x, y, 1, 0, 1.0f, 0.0f);  // Along axis 1

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::NORM_SLICE);
    REQUIRE(y.producer()->inputs().size() == 2);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

// Tests for utility operations
TEST_CASE("LogicalGraph Fill", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({16, 16}, "x", DataType::FP32);
    fill(x, 3.14f);

    REQUIRE(x.has_producer());
    REQUIRE(x.producer()->type() == OpType::FILL);
    REQUIRE(x.producer()->inputs().size() == 0);
    REQUIRE(x.producer()->outputs().size() == 1);
    REQUIRE(x.producer()->output() == &x);
    REQUIRE(std::holds_alternative<FillAttrs>(x.producer()->attrs()));

    auto attrs = std::get<FillAttrs>(x.producer()->attrs());
    REQUIRE(attrs.val == 3.14f);
}

TEST_CASE("LogicalGraph Copy", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = copy(x, "y");

    REQUIRE(y.shape() == x.shape());
    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::COPY);
    REQUIRE(y.producer()->inputs().size() == 1);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph Transpose", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = transpose(x, "y", 2.0f, 1);

    REQUIRE(y.shape() == std::vector<Index>({768, 32}));
    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::TRANSPOSE);
    REQUIRE(std::holds_alternative<TransposeAttrs>(y.producer()->attrs()));

    auto attrs = std::get<TransposeAttrs>(y.producer()->attrs());
    REQUIRE(attrs.alpha == 2.0f);
    REQUIRE(attrs.ndim == 1);
}

TEST_CASE("LogicalGraph Scatter", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = scatter(x, "y");

    REQUIRE(y.shape() == x.shape());
    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::SCATTER);
    REQUIRE(y.producer()->inputs().size() == 1);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph CopyIntersection", "[graph]")
{
    LogicalGraph g("test");

    auto& src = g.tensor({32, 768}, "src", DataType::FP32);
    auto& dst = g.tensor({16, 768}, "dst", DataType::FP32);
    copy_intersection(src, {0, 0}, dst, {0, 0});

    REQUIRE(dst.has_producer());
    REQUIRE(dst.producer()->type() == OpType::COPY_INTERSECTION);
    REQUIRE(dst.producer()->inputs().size() == 2);
    REQUIRE(dst.producer()->outputs().size() == 1);
    REQUIRE(dst.producer()->output() == &dst);
}

TEST_CASE("LogicalGraph ScaleFiber", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = g.tensor({32, 768}, "y", DataType::FP32);
    scale_fiber(x, y, 2.0f, 1, 0);  // Scale along axis 1

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::SCALE_FIBER);
    REQUIRE(y.producer()->inputs().size() == 2);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph ScaleSlice", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    auto& y = g.tensor({32, 768}, "y", DataType::FP32);
    scale_slice(x, y, 1.5f, 1);  // Scale along axis 1

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::SCALE_SLICE);
    REQUIRE(y.producer()->inputs().size() == 2);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
}

TEST_CASE("LogicalGraph Randn", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({32, 768}, "x", DataType::FP32);
    randn(x, {0, 0}, {32, 768}, 42, 0.0f, 1.0f);

    REQUIRE(x.has_producer());
    REQUIRE(x.producer()->type() == OpType::RANDN);
    REQUIRE(x.producer()->inputs().size() == 0);
    REQUIRE(x.producer()->outputs().size() == 1);
    REQUIRE(x.producer()->output() == &x);
    REQUIRE(std::holds_alternative<RandnAttrs>(x.producer()->attrs()));

    auto attrs = std::get<RandnAttrs>(x.producer()->attrs());
    REQUIRE(attrs.start == std::vector<Index>({0, 0}));
    REQUIRE(attrs.underlying_shape == std::vector<Index>({32, 768}));
    REQUIRE(attrs.seed == 42);
    REQUIRE(attrs.mean == 0.0f);
    REQUIRE(attrs.stddev == 1.0f);
}

// Tests for optimizer operations
TEST_CASE("LogicalGraph SgdStep", "[graph]")
{
    LogicalGraph g("test");

    auto& grad = g.tensor({32, 768}, "grad", DataType::FP32);
    auto& velocity = g.tensor({32, 768}, "velocity", DataType::FP32);
    auto& p = g.tensor({32, 768}, "p", DataType::FP32);

    sgd_step(100, 0.9f, 0.01f, 0.0001f, 0.0f, false, grad, velocity, p);

    REQUIRE(p.has_producer());
    REQUIRE(p.producer()->type() == OpType::SGD_STEP);
    REQUIRE(p.producer()->inputs().size() == 3);
    REQUIRE(p.producer()->outputs().size() == 1);
    REQUIRE(p.producer()->output() == &p);
    REQUIRE(std::holds_alternative<SgdStepAttrs>(p.producer()->attrs()));

    auto attrs = std::get<SgdStepAttrs>(p.producer()->attrs());
    REQUIRE(attrs.num_iter == 100);
    REQUIRE(attrs.momentum == 0.9f);
    REQUIRE(attrs.lr == 0.01f);
    REQUIRE(attrs.weight_decay == 0.0001f);
    REQUIRE(attrs.dampening == 0.0f);
    REQUIRE_FALSE(attrs.nesterov);
}

TEST_CASE("LogicalGraph AdamStep", "[graph]")
{
    LogicalGraph g("test");

    auto& grad = g.tensor({32, 768}, "grad", DataType::FP32);
    auto& first_moment = g.tensor({32, 768}, "first_moment", DataType::FP32);
    auto& second_moment = g.tensor({32, 768}, "second_moment", DataType::FP32);
    auto& p = g.tensor({32, 768}, "p", DataType::FP32);

    adam_step(100, 0.9f, 0.999f, 1e-8f, 0.001f, 0.01f, grad, first_moment, second_moment, p);

    REQUIRE(p.has_producer());
    REQUIRE(p.producer()->type() == OpType::ADAM_STEP);
    REQUIRE(p.producer()->inputs().size() == 4);
    REQUIRE(p.producer()->outputs().size() == 1);
    REQUIRE(p.producer()->output() == &p);
    REQUIRE(std::holds_alternative<AdamStepAttrs>(p.producer()->attrs()));

    auto attrs = std::get<AdamStepAttrs>(p.producer()->attrs());
    REQUIRE(attrs.num_iter == 100);
    REQUIRE(attrs.beta_1 == 0.9f);
    REQUIRE(attrs.beta_2 == 0.999f);
    REQUIRE(attrs.eps == 1e-8f);
    REQUIRE(attrs.lr == 0.001f);
    REQUIRE(attrs.weight_decay == 0.01f);
}

TEST_CASE("LogicalGraph AdamwStep", "[graph]")
{
    LogicalGraph g("test");

    auto& grad = g.tensor({32, 768}, "grad", DataType::FP32);
    auto& first_moment = g.tensor({32, 768}, "first_moment", DataType::FP32);
    auto& second_moment = g.tensor({32, 768}, "second_moment", DataType::FP32);
    auto& p = g.tensor({32, 768}, "p", DataType::FP32);

    adamw_step(100, 0.9f, 0.999f, 1e-8f, 0.001f, 0.01f, grad, first_moment, second_moment, p);

    REQUIRE(p.has_producer());
    REQUIRE(p.producer()->type() == OpType::ADAMW_STEP);
    REQUIRE(p.producer()->inputs().size() == 4);
    REQUIRE(p.producer()->outputs().size() == 1);
    REQUIRE(p.producer()->output() == &p);
    REQUIRE(std::holds_alternative<AdamStepAttrs>(p.producer()->attrs()));

    auto attrs = std::get<AdamStepAttrs>(p.producer()->attrs());
    REQUIRE(attrs.num_iter == 100);
    REQUIRE(attrs.beta_1 == 0.9f);
    REQUIRE(attrs.beta_2 == 0.999f);
    REQUIRE(attrs.eps == 1e-8f);
    REQUIRE(attrs.lr == 0.001f);
    REQUIRE(attrs.weight_decay == 0.01f);
}

// Tests for convolution operations
TEST_CASE("LogicalGraph Conv2dInplace", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({1, 28, 28}, "x", DataType::FP32);  // NCHW: [1, 1, 28, 28] but in WHCN format
    auto& c = g.tensor({3, 3}, "c", DataType::FP32);       // Kernel: [3, 3]
    auto& y = g.tensor({1, 26, 26}, "y", DataType::FP32);  // Output: [1, 26, 26]

    conv2d_inplace(x, c, y, 1.0f, 0.0f, {0, 0}, {1, 1}, {1, 1});

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::CONV2D_INPLACE);
    REQUIRE(y.producer()->inputs().size() == 3);
    REQUIRE(y.producer()->outputs().size() == 1);
    REQUIRE(y.producer()->output() == &y);
    REQUIRE(std::holds_alternative<Conv2dAttrs>(y.producer()->attrs()));

    auto attrs = std::get<Conv2dAttrs>(y.producer()->attrs());
    REQUIRE(attrs.alpha == 1.0f);
    REQUIRE(attrs.beta == 0.0f);
    REQUIRE(attrs.padding == std::array<Index, 2>({0, 0}));
    REQUIRE(attrs.stride == std::array<Index, 2>({1, 1}));
    REQUIRE(attrs.dilation == std::array<Index, 2>({1, 1}));
}

TEST_CASE("LogicalGraph Conv2dBwdInputInplace", "[graph]")
{
    LogicalGraph g("test");

    auto& dy = g.tensor({1, 26, 26}, "dy", DataType::FP32);
    auto& c = g.tensor({3, 3}, "c", DataType::FP32);
    auto& dx = g.tensor({1, 28, 28}, "dx", DataType::FP32);

    conv2d_bwd_input_inplace(dy, c, dx, 1.0f, 0.0f, {0, 0}, {1, 1}, {1, 1});

    REQUIRE(dx.has_producer());
    REQUIRE(dx.producer()->type() == OpType::CONV2D_BWD_INPUT_INPLACE);
    REQUIRE(dx.producer()->inputs().size() == 3);
    REQUIRE(dx.producer()->outputs().size() == 1);
    REQUIRE(dx.producer()->output() == &dx);
}

TEST_CASE("LogicalGraph Conv2dBwdWeightInplace", "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({1, 28, 28}, "x", DataType::FP32);
    auto& dy = g.tensor({1, 26, 26}, "dy", DataType::FP32);
    auto& dc = g.tensor({3, 3}, "dc", DataType::FP32);

    conv2d_bwd_weight_inplace(x, dy, dc, 1.0f, 0.0f, {0, 0}, {1, 1}, {1, 1});

    REQUIRE(dc.has_producer());
    REQUIRE(dc.producer()->type() == OpType::CONV2D_BWD_WEIGHT_INPLACE);
    REQUIRE(dc.producer()->inputs().size() == 3);
    REQUIRE(dc.producer()->outputs().size() == 1);
    REQUIRE(dc.producer()->output() == &dc);
}

// Tests for advanced operations
TEST_CASE("LogicalGraph FlashSdpaFwdCudnn", "[graph]")
{
    LogicalGraph g("test");

    auto& K = g.tensor({32, 64}, "K", DataType::FP32);
    auto& Q = g.tensor({32, 64}, "Q", DataType::FP32);
    auto& mask = g.tensor({32, 32}, "mask", DataType::FP32);
    auto& logsumexp = g.tensor({32}, "logsumexp", DataType::FP32);
    auto& V = g.tensor({32, 64}, "V", DataType::FP32);
    auto& A = g.tensor({32, 64}, "A", DataType::FP32);

    flash_sdpa_fwd_cudnn(K, Q, mask, logsumexp, V, A);

    REQUIRE(A.has_producer());
    REQUIRE(A.producer()->type() == OpType::FLASH_SDPA_FWD_CUDNN);
    REQUIRE(A.producer()->inputs().size() == 5);
    REQUIRE(A.producer()->outputs().size() == 1);
    REQUIRE(A.producer()->output() == &A);
}

TEST_CASE("LogicalGraph FlashSdpaBwdCudnn", "[graph]")
{
    LogicalGraph g("test");

    auto& K = g.tensor({32, 64}, "K", DataType::FP32);
    auto& Q = g.tensor({32, 64}, "Q", DataType::FP32);
    auto& V = g.tensor({32, 64}, "V", DataType::FP32);
    auto& A = g.tensor({32, 64}, "A", DataType::FP32);
    auto& dA = g.tensor({32, 64}, "dA", DataType::FP32);
    auto& mask = g.tensor({32, 32}, "mask", DataType::FP32);
    auto& logsumexp = g.tensor({32}, "logsumexp", DataType::FP32);
    auto& dK = g.tensor({32, 64}, "dK", DataType::FP32);
    auto& dQ = g.tensor({32, 64}, "dQ", DataType::FP32);
    auto& dV = g.tensor({32, 64}, "dV", DataType::FP32);

    flash_sdpa_bwd_cudnn(K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);

    REQUIRE(dK.has_producer());
    REQUIRE(dK.producer()->type() == OpType::FLASH_SDPA_BWD_CUDNN);
    REQUIRE(dK.producer()->inputs().size() == 10);
    REQUIRE(dK.producer()->outputs().size() == 3);
}

TEST_CASE("LogicalGraph Rope", "[graph]")
{
    LogicalGraph g("test");

    auto& sin_tensor = g.tensor({32, 64}, "sin", DataType::FP32);
    auto& cos_tensor = g.tensor({32, 64}, "cos", DataType::FP32);
    auto& src = g.tensor({32, 64}, "src", DataType::FP32);
    auto& dst = g.tensor({32, 64}, "dst", DataType::FP32);

    rope(sin_tensor, cos_tensor, src, dst);

    REQUIRE(dst.has_producer());
    REQUIRE(dst.producer()->type() == OpType::ROPE);
    REQUIRE(dst.producer()->inputs().size() == 3);
    REQUIRE(dst.producer()->outputs().size() == 1);
    REQUIRE(dst.producer()->output() == &dst);
}

TEST_CASE("LogicalGraph RopeBackward", "[graph]")
{
    LogicalGraph g("test");

    auto& sin_tensor = g.tensor({32, 64}, "sin", DataType::FP32);
    auto& cos_tensor = g.tensor({32, 64}, "cos", DataType::FP32);
    auto& dy = g.tensor({32, 64}, "dy", DataType::FP32);
    auto& dx = g.tensor({32, 64}, "dx", DataType::FP32);

    rope_backward(sin_tensor, cos_tensor, dy, dx);

    REQUIRE(dx.has_producer());
    REQUIRE(dx.producer()->type() == OpType::ROPE_BACKWARD);
    REQUIRE(dx.producer()->inputs().size() == 4);
    REQUIRE(dx.producer()->outputs().size() == 1);
    REQUIRE(dx.producer()->output() == &dx);
}
