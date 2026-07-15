/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/scatter.cc
 * Test TensorGraph scatter operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/scatter.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/scatter.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} 

TEST_CASE("TensorGraph scatter structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    nntile::TensorRef src = graph.data({5, 4});
    src->set_name("src");
    nntile::TensorRef dst = graph.data({5, 4});
    dst->set_name("dst");
    gt::scatter(src, dst);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SCATTER");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph scatter rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef src = graph.data({5, 4});
    src->set_name("src");
    nntile::TensorRef dst = graph.data({5, 4});
    dst->set_name("dst");

    REQUIRE_THROWS_AS(gt::scatter(nullptr, dst), std::invalid_argument);
    REQUIRE_THROWS_AS(gt::scatter(src, nullptr), std::invalid_argument);
}

TEST_CASE("TensorGraph scatter rejects shape mismatch", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef src = graph.data({5, 4});
    src->set_name("src");
    nntile::TensorRef dst = graph.data({3, 4});
    dst->set_name("dst");

    REQUIRE_THROWS_AS(gt::scatter(src, dst), std::invalid_argument);
}

// scatter requires src to be single-tiled; tiling all shared axes would
// violate that constraint, so no tiled-vs-untiled test is added here.
