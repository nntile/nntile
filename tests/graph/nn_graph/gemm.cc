/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/nn_graph/gemm.cc
 * Tests for NNGraph gemm autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

TEST_CASE("NNGraph Autograd Gemm ForwardAndBackward", "[graph][nn_graph]")
{
    NNGraph g("gemm");
    auto* a = g.tensor({2, 3}, "a", DataType::FP32, true);
    auto* b = g.tensor({3, 4}, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c");

    REQUIRE(c != nullptr);
    REQUIRE(c->has_producer());
    REQUIRE(c->shape() == (std::vector<Index>{2, 4}));

    auto* c_grad = g.get_or_create_grad(c, "c_grad");
    fill(Scalar(1.0), c_grad->data());
    c->backward();

    REQUIRE(a->has_grad());
    REQUIRE(b->has_grad());
    REQUIRE(a->grad()->shape() == (std::vector<Index>{2, 3}));
    REQUIRE(b->grad()->shape() == (std::vector<Index>{3, 4}));
}

TEST_CASE("NNGraph Autograd Gemm Backward TransposedA", "[graph][nn_graph]")
{
    SECTION("trans_b=false uses grad_C transpose for grad_A")
    {
        NNGraph g("gemm_transposed_a");
        auto* a = g.tensor({4, 2}, "a", DataType::FP32, true);
        auto* b = g.tensor({4, 3}, "b", DataType::FP32, true);
        auto* c = gemm(a, b, "c", Scalar(1.0), true, false);

        REQUIRE(c != nullptr);
        REQUIRE(c->shape() == (std::vector<Index>{2, 3}));

        auto* c_grad = g.get_or_create_grad(c, "c_grad");
        fill(Scalar(1.0), c_grad->data());

        REQUIRE_NOTHROW(c->backward());
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{4, 2}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{4, 3}));
    }

    SECTION("trans_b=true uses grad_C transpose for grad_A")
    {
        NNGraph g("gemm_transposed_a_transposed_b");
        auto* a = g.tensor({4, 2}, "a", DataType::FP32, true);
        auto* b = g.tensor({3, 4}, "b", DataType::FP32, true);
        auto* c = gemm(a, b, "c", Scalar(1.0), true, true);

        REQUIRE(c != nullptr);
        REQUIRE(c->shape() == (std::vector<Index>{2, 3}));

        auto* c_grad = g.get_or_create_grad(c, "c_grad");
        fill(Scalar(1.0), c_grad->data());

        REQUIRE_NOTHROW(c->backward());
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{4, 2}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{3, 4}));
    }
}
