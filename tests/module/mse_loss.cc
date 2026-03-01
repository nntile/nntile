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

