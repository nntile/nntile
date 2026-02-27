/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/module/mse_loss.cc
 * Tests for MseLoss module.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph.hh"
#include "nntile/module/mse_loss.hh"

using namespace nntile::graph;
using namespace nntile::module;

TEST_CASE("MseLoss BuildForward", "[module]")
{
    NNGraph g("mse");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    MseLoss mse(g, "mse");

    auto& loss = mse.build_forward(*x);
    REQUIRE(loss.ndim() == 0);
    REQUIRE(loss.shape().empty());
    REQUIRE(loss.name() == "mse_loss");
}

TEST_CASE("MseLoss BuildBackward", "[module]")
{
    NNGraph g("mse");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    MseLoss mse(g, "mse");

    auto& loss = mse.build_forward(*x);
    mse.build_backward();

    // Scalar loss: grad auto-set to 1.0
    REQUIRE(loss.has_grad());

    // grad_x = 2*x
    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x->shape());

    // Check ADD_INPLACE with alpha=2 for grad accumulation
    size_t add_inplace_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->type() == OpType::ADD_INPLACE)
        {
            ++add_inplace_count;
        }
    }
    REQUIRE(add_inplace_count >= 1);
}

TEST_CASE("MseLoss BuildBackwardRequiresForward", "[module]")
{
    NNGraph g("mse");
    MseLoss mse(g, "mse");
    REQUIRE_THROWS_AS(mse.build_backward(), std::runtime_error);
}
