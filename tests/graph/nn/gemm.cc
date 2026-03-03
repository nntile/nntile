/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/gemm.cc
 * Test NNGraph gemm autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

namespace
{

constexpr Scalar gemm_alpha_one = 1.0;
constexpr bool trans_a_default = false;
constexpr bool trans_b_default = false;
constexpr Index ndim_one = 1;
constexpr Index batch_ndim_none = 0;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm structure", "[graph][nn_graph]")
{
    const auto [M, K, N] = GENERATE(
        std::tuple{Index(2), Index(3), Index(4)},
        std::tuple{Index(3), Index(4), Index(3)});

    NNGraph g("gemm_structure");
    auto* a = g.tensor({M, K}, "a", DataType::FP32);
    auto* b = g.tensor({K, N}, "b", DataType::FP32);
    auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a_default, trans_b_default,
                   ndim_one, batch_ndim_none);

    REQUIRE(c != nullptr);
    REQUIRE(c->has_producer());
    REQUIRE(c->shape() == (std::vector<Index>{M, N}));
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "GEMM");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward", "[graph][nn_graph]")
{
    const auto [M, K, N, grad_fill_val] = GENERATE(
        std::tuple{Index(2), Index(3), Index(4), Scalar(1.0)},
        std::tuple{Index(3), Index(4), Index(3), Scalar(-1.0)});

    NNGraph g("gemm_backward");
    auto* a = g.tensor({M, K}, "a", DataType::FP32);
    auto* b = g.tensor({K, N}, "b", DataType::FP32);
    auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a_default, trans_b_default,
                   ndim_one, batch_ndim_none);

    auto* c_grad = g.get_or_create_grad(c, "c_grad");
    fill(grad_fill_val, c_grad->data());
    c->backward();

    REQUIRE(a->has_grad());
    REQUIRE(b->has_grad());
    REQUIRE(a->grad()->shape() == (std::vector<Index>{M, K}));
    REQUIRE(b->grad()->shape() == (std::vector<Index>{K, N}));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward and backward", "[graph][nn_graph]")
{
    const auto [M, K, N, gemm_alpha, grad_fill_val] = GENERATE(
        std::tuple{Index(2), Index(3), Index(4), Scalar(1.0), Scalar(1.0)},
        std::tuple{Index(3), Index(4), Index(3), Scalar(0.5), Scalar(1.0)},
        std::tuple{Index(4), Index(5), Index(6), Scalar(2.0), Scalar(-1.0)});

    NNGraph g("gemm");
    auto* a = g.tensor({M, K}, "a", DataType::FP32, true);
    auto* b = g.tensor({K, N}, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c", gemm_alpha, trans_a_default, trans_b_default,
                   ndim_one, batch_ndim_none);

    REQUIRE(c != nullptr);
    REQUIRE(c->has_producer());
    REQUIRE(c->shape() == (std::vector<Index>{M, N}));

    auto* c_grad = g.get_or_create_grad(c, "c_grad");
    fill(grad_fill_val, c_grad->data());
    c->backward();

    REQUIRE(a->has_grad());
    REQUIRE(b->has_grad());
    REQUIRE(a->grad()->shape() == (std::vector<Index>{M, K}));
    REQUIRE(b->grad()->shape() == (std::vector<Index>{K, N}));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward transposed A", "[graph][nn_graph]")
{
    constexpr Scalar grad_fill_val = 1.0;

    SECTION("trans_b=false uses grad_C transpose for grad_A")
    {
        constexpr bool trans_a = true;
        constexpr bool trans_b = false;

        NNGraph g("gemm_transposed_a");
        auto* a = g.tensor({4, 2}, "a", DataType::FP32, true);
        auto* b = g.tensor({4, 3}, "b", DataType::FP32, true);
        auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a, trans_b,
                       ndim_one, batch_ndim_none);

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
        constexpr bool trans_a = true;
        constexpr bool trans_b = true;

        NNGraph g("gemm_transposed_a_transposed_b");
        auto* a = g.tensor({4, 2}, "a", DataType::FP32, true);
        auto* b = g.tensor({3, 4}, "b", DataType::FP32, true);
        auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a, trans_b,
                       ndim_one, batch_ndim_none);

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
