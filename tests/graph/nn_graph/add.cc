/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/nn_graph/add.cc
 * Tests for NNGraph add autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

TEST_CASE("NNGraph Autograd Add Callable", "[graph][nn_graph]")
{
    NNGraph g("add_callable");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);
    Add add_fn;
    auto outs = add_fn(1.0, x, 1.0, y, "z");
    auto* z = outs.empty() ? nullptr : outs[0];
    REQUIRE(z != nullptr);
    REQUIRE(z->has_producer());
}

TEST_CASE("NNGraph Autograd Add Backward", "[graph][nn_graph]")
{
    NNGraph g("autograd_add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);
    Scalar alpha = 2.0;
    Scalar beta = 3.0;

    auto* z = add(alpha, x, beta, y, "z");

    REQUIRE(z->has_producer());
    REQUIRE_FALSE(z->is_leaf());
    REQUIRE(x->is_leaf());
    REQUIRE(y->is_leaf());

    auto* z_grad = g.get_or_create_grad(z, "z_grad");
    fill(Scalar(1.0), z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());

    size_t add_inplace_count = 0;
    for(const auto& op : g.logical_graph().ops())
    {
        if(op->type() == OpType::ADD_INPLACE)
            ++add_inplace_count;
    }
    REQUIRE(add_inplace_count == 2);
}

TEST_CASE("NNGraph Autograd Add Chain", "[graph][nn_graph]")
{
    NNGraph g("add_chain");
    auto* x = g.tensor({2, 2}, "x", DataType::FP32);
    auto* y = g.tensor({2, 2}, "y", DataType::FP32);
    auto* u = g.tensor({2, 2}, "u", DataType::FP32);

    auto* w = add(Scalar(1.0), x, Scalar(1.0), y, "w");
    auto* z = add(Scalar(1.0), w, Scalar(1.0), u, "z");

    auto* z_grad = g.get_or_create_grad(z, "z_grad");
    fill(Scalar(1.0), z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
    REQUIRE(u->has_grad());
    REQUIRE(w->has_grad());
}

TEST_CASE("NNGraph Autograd Add Diamond", "[graph][nn_graph]")
{
    NNGraph g("add_diamond");
    auto* x = g.tensor({2, 2}, "x", DataType::FP32);
    auto* y = g.tensor({2, 2}, "y", DataType::FP32);

    auto* w = add(Scalar(1.0), x, Scalar(1.0), y, "w");
    auto* v = add(Scalar(1.0), w, Scalar(1.0), y, "v");
    auto* z = add(Scalar(1.0), v, Scalar(1.0), w, "z");

    auto* z_grad = g.get_or_create_grad(z, "z_grad");
    fill(Scalar(1.0), z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
    REQUIRE(w->has_grad());
    REQUIRE(v->has_grad());
}

TEST_CASE("NNGraph Autograd Add ForwardAndBackward", "[graph][nn_graph]")
{
    NNGraph g("add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32, true);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32, true);
    auto* z = add(Scalar(1.0), x, Scalar(1.0), y, "z");

    REQUIRE(z != nullptr);
    REQUIRE(z->has_producer());
    REQUIRE(z->shape() == (std::vector<Index>{2, 3}));

    auto* z_grad = g.get_or_create_grad(z, "z_grad");
    fill(Scalar(1.0), z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
}
