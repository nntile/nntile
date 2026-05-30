/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/randn.cc
 * Test TensorGraph randn operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/randn.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/randn.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr unsigned long long seed = 42;
constexpr Scalar mean = 0.0;
constexpr Scalar stddev = 1.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} 

TEST_CASE("TensorGraph randn structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *dst = graph.data({4, 5})->set_name("dst");
    gt::randn(dst, {0, 0}, {4, 5}, seed, mean, stddev);

    REQUIRE(graph.num_data() == 1);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "RANDN");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph randn rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");

    REQUIRE_THROWS_AS(gt::randn(nullptr, {0, 0}, {4, 5}, seed, mean, stddev),
        std::invalid_argument);
}

TEST_CASE("TensorGraph randn rejects mismatched start/underlying_shape",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *dst = graph.data({4, 5})->set_name("dst");

    REQUIRE_THROWS_AS(gt::randn(dst, {0}, {4, 5}, seed, mean, stddev),
        std::invalid_argument);
    REQUIRE_THROWS_AS(gt::randn(dst, {0, 0}, {4}, seed, mean, stddev),
        std::invalid_argument);
}

