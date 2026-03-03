/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/add.cc
 * Test NNGraph add autograd operation.
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
    "NNGraph add structure", "[graph][nn_graph]")
{
    const auto [alpha, beta] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(3.0)},
        std::tuple{Scalar(0.5), Scalar(-1.0)});

    constexpr Index dim0 = 2;
    constexpr Index dim1 = 3;

    NNGraph g("add_structure");
    auto* x = g.tensor({dim0, dim1}, "x", DataType::FP32);
    auto* y = g.tensor({dim0, dim1}, "y", DataType::FP32);
    auto* z = add(alpha, x, beta, y, "z");

    REQUIRE(z != nullptr);
    REQUIRE(z->has_producer());
    REQUIRE(z->shape() == (std::vector<Index>{dim0, dim1}));
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "ADD");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add backward", "[graph][nn_graph]")
{
    const auto [alpha, beta, grad_fill_val] = GENERATE(
        std::tuple{Scalar(2.0), Scalar(3.0), Scalar(1.0)},
        std::tuple{Scalar(0.5), Scalar(-1.0), Scalar(2.0)},
        std::tuple{Scalar(1.0), Scalar(0.0), Scalar(1.0)});

    NNGraph g("autograd_add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);

    auto* z = add(alpha, x, beta, y, "z");

    REQUIRE(z->has_producer());
    REQUIRE_FALSE(z->is_leaf());
    REQUIRE(x->is_leaf());
    REQUIRE(y->is_leaf());

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());

    size_t add_inplace_count = 0;
    for(const auto& op : g.tensor_graph().ops())
    {
        if(op->op_name() == "ADD_INPLACE")
            ++add_inplace_count;
    }
    REQUIRE(add_inplace_count == 2);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add chain", "[graph][nn_graph]")
{
    const auto [add_alpha, add_beta, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(0.5), Scalar(2.0), Scalar(-1.0)});

    NNGraph g("add_chain");
    auto* x = g.tensor({2, 2}, "x", DataType::FP32);
    auto* y = g.tensor({2, 2}, "y", DataType::FP32);
    auto* u = g.tensor({2, 2}, "u", DataType::FP32);

    auto* w = add(add_alpha, x, add_beta, y, "w");
    auto* z = add(add_alpha, w, add_beta, u, "z");

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
    REQUIRE(u->has_grad());
    REQUIRE(w->has_grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add diamond", "[graph][nn_graph]")
{
    const auto [add_alpha, add_beta, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(0.5), Scalar(1.0)});

    NNGraph g("add_diamond");
    auto* x = g.tensor({2, 2}, "x", DataType::FP32);
    auto* y = g.tensor({2, 2}, "y", DataType::FP32);

    auto* w = add(add_alpha, x, add_beta, y, "w");
    auto* v = add(add_alpha, w, add_beta, y, "v");
    auto* z = add(add_alpha, v, add_beta, w, "z");

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
    REQUIRE(w->has_grad());
    REQUIRE(v->has_grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add forward and backward", "[graph][nn_graph]")
{
    const auto [add_alpha, add_beta, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(3.0), Scalar(1.0)},
        std::tuple{Scalar(0.5), Scalar(-1.0), Scalar(2.0)});

    NNGraph g("add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32, true);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32, true);
    auto* z = add(add_alpha, x, add_beta, y, "z");

    REQUIRE(z != nullptr);
    REQUIRE(z->has_producer());
    REQUIRE(z->shape() == (std::vector<Index>{2, 3}));

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
}
