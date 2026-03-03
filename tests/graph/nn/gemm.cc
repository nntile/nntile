/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/nn/gemm.cc
 * Tests for NNGraph gemm autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph Autograd Gemm ForwardAndBackward", "[graph][nn_graph]")
{
    const Scalar gemm_alpha = GENERATE(Scalar(1.0));
    const bool trans_a = GENERATE(false);
    const bool trans_b = GENERATE(false);
    const Index ndim = GENERATE(Index(1));
    const Index batch_ndim = GENERATE(Index(0));
    const Scalar grad_fill_val = GENERATE(Scalar(1.0));

    NNGraph g("gemm");
    auto* a = g.tensor({2, 3}, "a", DataType::FP32, true);
    auto* b = g.tensor({3, 4}, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c", gemm_alpha, trans_a, trans_b, ndim, batch_ndim);

    REQUIRE(c != nullptr);
    REQUIRE(c->has_producer());
    REQUIRE(c->shape() == (std::vector<Index>{2, 4}));

    auto* c_grad = g.get_or_create_grad(c, "c_grad");
    fill(grad_fill_val, c_grad->data());
    c->backward();

    REQUIRE(a->has_grad());
    REQUIRE(b->has_grad());
    REQUIRE(a->grad()->shape() == (std::vector<Index>{2, 3}));
    REQUIRE(b->grad()->shape() == (std::vector<Index>{3, 4}));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph Autograd Gemm Backward TransposedA", "[graph][nn_graph]")
{
    SECTION("trans_b=false uses grad_C transpose for grad_A")
    {
        const Scalar gemm_alpha = GENERATE(Scalar(1.0));
        const bool trans_a = GENERATE(true);
        const bool trans_b = GENERATE(false);
        const Index ndim = GENERATE(Index(1));
        const Index batch_ndim = GENERATE(Index(0));
        const Scalar grad_fill_val = GENERATE(Scalar(1.0));

        NNGraph g("gemm_transposed_a");
        auto* a = g.tensor({4, 2}, "a", DataType::FP32, true);
        auto* b = g.tensor({4, 3}, "b", DataType::FP32, true);
        auto* c = gemm(a, b, "c", gemm_alpha, trans_a, trans_b, ndim, batch_ndim);

        REQUIRE(c != nullptr);
        REQUIRE(c->shape() == (std::vector<Index>{2, 3}));

        auto* c_grad = g.get_or_create_grad(c, "c_grad");
        fill(grad_fill_val, c_grad->data());

        REQUIRE_NOTHROW(c->backward());
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{4, 2}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{4, 3}));
    }

    SECTION("trans_b=true uses grad_C transpose for grad_A")
    {
        const Scalar gemm_alpha = GENERATE(Scalar(1.0));
        const bool trans_a = GENERATE(true);
        const bool trans_b = GENERATE(true);
        const Index ndim = GENERATE(Index(1));
        const Index batch_ndim = GENERATE(Index(0));
        const Scalar grad_fill_val = GENERATE(Scalar(1.0));

        NNGraph g("gemm_transposed_a_transposed_b");
        auto* a = g.tensor({4, 2}, "a", DataType::FP32, true);
        auto* b = g.tensor({3, 4}, "b", DataType::FP32, true);
        auto* c = gemm(a, b, "c", gemm_alpha, trans_a, trans_b, ndim, batch_ndim);

        REQUIRE(c != nullptr);
        REQUIRE(c->shape() == (std::vector<Index>{2, 3}));

        auto* c_grad = g.get_or_create_grad(c, "c_grad");
        fill(grad_fill_val, c_grad->data());

        REQUIRE_NOTHROW(c->backward());
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{4, 2}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{3, 4}));
    }
}
