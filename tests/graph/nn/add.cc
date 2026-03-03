/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/nn/add.cc
 * Tests for NNGraph add autograd operation.
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
    "NNGraph Autograd Add build_forward", "[graph][nn_graph]")
{
    const Scalar alpha = GENERATE(Scalar(1.0));
    const Scalar beta = GENERATE(Scalar(1.0));

    NNGraph g("add_build_forward");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);
    auto* z = add(alpha, x, beta, y, "z");
    REQUIRE(z != nullptr);
    REQUIRE(z->has_producer());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph Autograd Add Backward", "[graph][nn_graph]")
{
    const Scalar alpha = GENERATE(Scalar(2.0));
    const Scalar beta = GENERATE(Scalar(3.0));
    const Scalar grad_fill_val = GENERATE(Scalar(1.0));

    NNGraph g("autograd_add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);

    auto* z = add(alpha, x, beta, y, "z");

    REQUIRE(z->has_producer());
    REQUIRE_FALSE(z->is_leaf());
    REQUIRE(x->is_leaf());
    REQUIRE(y->is_leaf());

    auto* z_grad = g.get_or_create_grad(z, "z_grad");
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
    "NNGraph Autograd Add Chain", "[graph][nn_graph]")
{
    const Scalar add_alpha = GENERATE(Scalar(1.0));
    const Scalar add_beta = GENERATE(Scalar(1.0));
    const Scalar grad_fill_val = GENERATE(Scalar(1.0));

    NNGraph g("add_chain");
    auto* x = g.tensor({2, 2}, "x", DataType::FP32);
    auto* y = g.tensor({2, 2}, "y", DataType::FP32);
    auto* u = g.tensor({2, 2}, "u", DataType::FP32);

    auto* w = add(add_alpha, x, add_beta, y, "w");
    auto* z = add(add_alpha, w, add_beta, u, "z");

    auto* z_grad = g.get_or_create_grad(z, "z_grad");
    fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
    REQUIRE(u->has_grad());
    REQUIRE(w->has_grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph Autograd Add Diamond", "[graph][nn_graph]")
{
    const Scalar add_alpha = GENERATE(Scalar(1.0));
    const Scalar add_beta = GENERATE(Scalar(1.0));
    const Scalar grad_fill_val = GENERATE(Scalar(1.0));

    NNGraph g("add_diamond");
    auto* x = g.tensor({2, 2}, "x", DataType::FP32);
    auto* y = g.tensor({2, 2}, "y", DataType::FP32);

    auto* w = add(add_alpha, x, add_beta, y, "w");
    auto* v = add(add_alpha, w, add_beta, y, "v");
    auto* z = add(add_alpha, v, add_beta, w, "z");

    auto* z_grad = g.get_or_create_grad(z, "z_grad");
    fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
    REQUIRE(w->has_grad());
    REQUIRE(v->has_grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph Autograd Add ForwardAndBackward", "[graph][nn_graph]")
{
    const Scalar add_alpha = GENERATE(Scalar(1.0));
    const Scalar add_beta = GENERATE(Scalar(1.0));
    const Scalar grad_fill_val = GENERATE(Scalar(1.0));

    NNGraph g("add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32, true);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32, true);
    auto* z = add(add_alpha, x, add_beta, y, "z");

    REQUIRE(z != nullptr);
    REQUIRE(z->has_producer());
    REQUIRE(z->shape() == (std::vector<Index>{2, 3}));

    auto* z_grad = g.get_or_create_grad(z, "z_grad");
    fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
}
