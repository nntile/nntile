/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/module/activation.cc
 * Tests for Activation module.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <stdexcept>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/module/activation.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::module;

TEST_CASE("Activation AllTypes", "[module]")
{
    NNGraph g("activation");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);

    Activation gelu(&g, "gelu", ActivationType::GELU);
    Activation gelutanh(&g, "gelutanh", ActivationType::GELUTANH);
    Activation relu(&g, "relu", ActivationType::RELU);
    Activation silu(&g, "silu", ActivationType::SILU);

    auto* out_gelu = gelu.forward(input);
    REQUIRE(out_gelu->shape() == std::vector<Index>({2, 3}));

    REQUIRE(gelu.type() == ActivationType::GELU);
    REQUIRE(gelutanh.type() == ActivationType::GELUTANH);
    REQUIRE(relu.type() == ActivationType::RELU);
    REQUIRE(silu.type() == ActivationType::SILU);
}

TEST_CASE("Activation TypeFromString", "[module]")
{
    REQUIRE(activation_type_from_string("gelu") == ActivationType::GELU);
    REQUIRE(activation_type_from_string("gelutanh") == ActivationType::GELUTANH);
    REQUIRE(activation_type_from_string("relu") == ActivationType::RELU);
    REQUIRE(activation_type_from_string("silu") == ActivationType::SILU);

    REQUIRE_THROWS_AS(activation_type_from_string("unknown"),
                      std::invalid_argument);
}
